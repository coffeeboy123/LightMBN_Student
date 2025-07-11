import random
import numpy as np
import torch
from utils.functions import evaluation
from utils.re_ranking import re_ranking, re_ranking_gpu

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int):
    """실험 재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Engine:
    def __init__(self, args, model, optimizer, scheduler, loss, loader, ckpt):
        # ─── 재현성: 시드 고정 ────────────────────────────────
        if hasattr(args, 'seed'):
            set_seed(args.seed)
        # ──────────────────────────────────────────────────────

        self.args = args
        self.train_loader = loader.train_loader
        self.validation_loader = loader.validation_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.galleryset
        self.queryset = loader.queryset

        self.ckpt = ckpt
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

        self.lr = 0.0
        self.device = torch.device("cpu" if args.cpu else "cuda")

        self.train_ce_loss_history = []
        self.train_ms_loss_history = []
        self.val_ce_loss_history = []
        self.val_ms_loss_history = []

        if torch.cuda.is_available():
            self.ckpt.write_log("[INFO] GPU: " + torch.cuda.get_device_name(0))

        self.ckpt.write_log(
            "[INFO] Starting from epoch {}".format(self.scheduler.last_epoch + 1)
        )

        if args.wandb and wandb is not None:
            self.wandb = True
            wandb.init(project=args.wandb_name)
        else:
            self.wandb = False

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.ckpt.dir)

    def train(self):
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_last_lr()[0]

        if lr != self.lr:
            self.ckpt.write_log(
                "[INFO] Epoch: {}\tLearning rate: {:.2e}".format(epoch + 1, lr)
            )
            self.lr = lr
        self.loss.start_log()
        running_loss = 0.0
        ce_running_loss = 0.0
        ms_running_loss = 0.0

        self.model.train()

        for batch, d in enumerate(self.train_loader):
            inputs, labels = self._parse_data_for_train(d)

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            total_loss, ce_loss, ms_loss = self.loss.compute(outputs, labels)

            running_loss += total_loss.item()
            if ce_loss is not None:
                ce_running_loss += ce_loss
            if ms_loss is not None:
                ms_running_loss += ms_loss

            total_loss.backward()
            self.optimizer.step()

            self.ckpt.write_log(
                "\r[INFO] [{}/{}]\t{}/{}\t{}".format(
                    epoch + 1,
                    self.args.epochs,
                    batch + 1,
                    len(self.train_loader),
                    self.loss.display_loss(batch),
                ),
                end="" if batch + 1 != len(self.train_loader) else "\n",
            )

            if self.wandb and wandb is not None:
                wandb.log(self.loss.get_loss_dict(batch))

        self.scheduler.step()
        self.loss.end_log(len(self.train_loader))

        avg_loss = running_loss / len(self.train_loader)
        avg_ce = ce_running_loss / len(self.train_loader) if ce_running_loss != 0 else None
        avg_ms = ms_running_loss / len(self.train_loader) if ms_running_loss != 0 else None

        # checkpoint에 기록
        self.ckpt.loss_history.append(avg_loss)
        self.train_ce_loss_history.append(avg_ce)
        self.train_ms_loss_history.append(avg_ms)

        self.writer.add_scalar('Loss/Train_Total', avg_loss, epoch + 1)
        self.writer.add_scalar('Loss/Train_CE', avg_ce, epoch + 1)
        self.writer.add_scalar('Loss/Train_MS', avg_ms, epoch + 1)

        self._last_train_loss = avg_loss
        self._last_train_ce = avg_ce
        self._last_train_ms = avg_ms



        # loss curve 저장 (원하는 시점에—예: 매 epoch마다)

    def validation(self):
        epoch = self.scheduler.last_epoch
        running_loss = 0.0
        ce_running_loss = 0.0
        ms_running_loss = 0.0
        self.model.eval()

        with torch.no_grad():  # <- 반드시 감싸주세요!
            for batch, d in enumerate(self.validation_loader):
                inputs, labels = self._parse_data_for_train(d)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                total_loss, ce_loss, ms_loss = self.loss.compute(outputs, labels)
                running_loss += total_loss.item()
                if ce_loss is not None:
                    ce_running_loss += ce_loss
                if ms_loss is not None:
                    ms_running_loss += ms_loss


                self.ckpt.write_log(
                    "\r[VAL] [{}/{}]\t{}/{}\t{}".format(
                        epoch,
                        self.args.epochs,
                        batch + 1,
                        len(self.validation_loader),
                        self.loss.display_loss(batch),
                    ),
                    end="" if batch + 1 != len(self.validation_loader) else "\n",
                )

                if self.wandb and wandb is not None:
                    wandb.log({"val_loss_step": loss.item()})

        avg_loss = running_loss / len(self.validation_loader)
        avg_ce = ce_running_loss / len(self.validation_loader) if ce_running_loss != 0 else None
        avg_ms = ms_running_loss / len(self.validation_loader) if ms_running_loss != 0 else None

        self.ckpt.validation_loss_history.append(avg_loss)
        self.val_ce_loss_history.append(avg_ce)
        self.val_ms_loss_history.append(avg_ms)

        self.ckpt.plot_losses(
            self.ckpt.loss_history,  # total train loss
            self.ckpt.validation_loss_history,  # total val loss
            self.train_ce_loss_history,  # ce train loss
            self.val_ce_loss_history,  # ce val loss
            self.train_ms_loss_history,  # ms train loss
            self.val_ms_loss_history,  # ms val loss
        )

        self.writer.add_scalar('Loss/Val_Total', avg_loss, epoch + 1)
        self.writer.add_scalar('Loss/Val_CE', avg_ce, epoch + 1)
        self.writer.add_scalar('Loss/Val_MS', avg_ms, epoch + 1)

        self.writer.add_scalars('Loss/Total', {
            'Train': self._last_train_loss,
            'Val': avg_loss
        }, epoch + 1)
        self.writer.add_scalars('Loss/CE', {
            'Train': self._last_train_ce,
            'Val': avg_ce
        }, epoch + 1)
        self.writer.add_scalars('Loss/MS', {
            'Train': self._last_train_ms,
            'Val': avg_ms
        }, epoch + 1)

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckpt.write_log("\n[INFO] Test:")
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 6))

        with torch.no_grad():
            qf, query_ids, query_cams = self.extract_feature(
                self.query_loader, self.args
            )
            gf, gallery_ids, gallery_cams = self.extract_feature(
                self.test_loader, self.args
            )

        if self.args.re_rank:
            dist = re_ranking_gpu(qf, gf, 20, 6, 0.3)
        else:
            dist = 1 - torch.mm(qf, gf.t()).cpu().numpy()

        r, m_ap = evaluation(
            dist, query_ids, gallery_ids, query_cams, gallery_cams, 50
        )

        # (이하 생략 — 기존 코드 유지)
        ...

    # (나머지 메서드도 원본과 동일)


        self.ckpt.log[-1, 0] = epoch
        self.ckpt.log[-1, 1] = m_ap
        self.ckpt.log[-1, 2] = r[0]
        self.ckpt.log[-1, 3] = r[2]
        self.ckpt.log[-1, 4] = r[4]
        self.ckpt.log[-1, 5] = r[9]

        # ✅ 이전 기록만으로 best rank1/mAP 확인
        previous_log = self.ckpt.log[:-1, :]  # exclude current
        if previous_log.shape[0] > 0:
            # 최대 rank1을 가지는 모든 인덱스 추출
            best_rank1_value = previous_log[:, 2].max().item()
            rank1_mask = previous_log[:, 2] == best_rank1_value
            best_map_value = previous_log[rank1_mask, 1].max().item()

            best_epoch = int(previous_log[(rank1_mask) & (previous_log[:, 1] == best_map_value), 0][0].item())
        else:
            best_rank1_value = 0
            best_map_value = 0
            best_epoch = -1

        # ✅ best 조건 판별
        is_best = False
        if r[0] > best_rank1_value:
            is_best = True
        elif r[0] == best_rank1_value and m_ap > best_map_value:
            is_best = True

        # ✅ 현재 에포크가 best면 로그 정보도 갱신
        if is_best:
            best_rank1_value = r[0]
            best_map_value = m_ap
            best_epoch = epoch

        self.ckpt.write_log(
            "[INFO] mAP: {:.4f} | rank1: {:.4f} | rank3: {:.4f} | rank5: {:.4f} | rank10: {:.4f} "
            "(Best rank1: {:.4f} @epoch {})".format(
                m_ap, r[0], r[2], r[4], r[9], best_rank1_value, best_epoch
            ),
            refresh=True,
        )

        if not self.args.test_only:
            self._save_checkpoint(
                epoch,
                r[0],
                self.ckpt.dir,
                is_best=is_best,
            )

        if self.wandb is True and wandb is not None:
            wandb.log(
                {
                    "mAP": m_ap,
                    "rank1": r[0],
                    "rank3": r[2],
                    "rank5": r[4],
                    "rank10": r[9],
                }
            )
        # TensorBoard 기록 추가
        self.writer.add_scalars('Eval/Rank', {
            'rank1': r[0],
            'rank3': r[2],
            'rank5': r[4],
            'rank10': r[9],
        }, epoch + 1)
        self.writer.add_scalar('Eval/mAP', m_ap, epoch + 1)

        self.ckpt.plot_map_rank(epoch + 1)


    def close(self):
        self.writer.close()


    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3) - 1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    def extract_feature(self, loader, args):
        features = torch.FloatTensor()
        pids, camids = [], []

        for d in loader:
            inputs, pid, camid = self._parse_data_for_eval(d)
            input_img = inputs.to(self.device)
            outputs = self.model(input_img)

            f1 = outputs[2].data.cpu()
            # flip
            inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
            input_img = inputs.to(self.device)
            outputs = self.model(input_img)
            f2 = outputs[2].data.cpu()

            ff = f1 + f2
            if ff.dim() == 3:
                fnorm = torch.norm(
                    ff, p=2, dim=1, keepdim=True
                )  # * np.sqrt(ff.shape[2])
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)

            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
            pids.extend(pid)
            camids.extend(camid)

        return features, np.asarray(pids), np.asarray(camids)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1

            return epoch > self.args.epochs

    # tools for reid datamanager data_v2
    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        self.ckpt.save_checkpoint(
            {
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "rank1": rank1,
                "optimizer": self.optimizer.state_dict(),
                "log": self.ckpt.log,
                # 'scheduler': self.scheduler.state_dict(),
            },
            save_dir,
            is_best=is_best,
        )
