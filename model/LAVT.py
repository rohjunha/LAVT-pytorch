import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import transformers
from deepspeed.ops.adam import DeepSpeedCPUAdam

from model.segmentation import Segmentation
from model.swin_transformer import build_model
from utils.poly_lr_decay import PolynomialLRDecay
from utils.util import load_pretrained_swin


class LAVT(pl.LightningModule):
    def __init__(self, config, args):
        pl.LightningModule.__init__(self)
        self.config = config
        self.args = args
        self.textEncoder = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.imageEncoder = build_model(self.config)
        load_pretrained_swin(self.config, self.imageEncoder)
        self.Seg = Segmentation()

    def forward(self, img, emb, att_mask):
        _, _, H, _ = img.size()
        hidden_state = self.textEncoder(emb, attention_mask=att_mask)[0]
        fuse_feats = self.imageEncoder(img, hidden_state)
        pred = self.Seg(fuse_feats)
        _, _, h, _ = pred.size()
        assert H % h == 0

        return F.interpolate(pred, scale_factor=int(H // h), mode='bilinear', align_corners=True)

    def configure_optimizers(self):
        opt_cls = AdamW  #DeepSpeedCPUAdam
        optimizer = opt_cls(self.parameters(),
                            lr=self.args.lr,
                            weight_decay=self.args.weight_decay)
        scheduler = PolynomialLRDecay(optimizer,
                                      max_decay_steps=self.args.max_decay_steps,
                                      end_learning_rate=self.args.end_lr,
                                      power=self.args.power)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img, target, emb, att_mask = batch

        emb = emb.squeeze(1)
        att_mask = att_mask.squeeze(1)

        output = self.forward(img, emb, att_mask)
        return F.cross_entropy(output, target)

    def validation_step(self, batch, batch_idx):
        img, target, emb, att_mask = batch

        batch_size = img.size(0)
        emb = emb.squeeze(1)
        att_mask = att_mask.squeeze(1)

        _, o_H, o_W = target.size()

        # compute output for different mode
        if self.args.eval_mode == 'cat':
            emb = emb.view(batch_size, -1)
            att_mask = att_mask.view(batch_size, -1)
            output = self.forward(img, emb, att_mask)

        if self.args.eval_mode == 'avg':
            _, _, num_of_sent = emb.size()
            outputs = []
            for s in range(num_of_sent):
                emb_s, att_mask_s = emb[:, :, s], att_mask[:, :, s]
                outputs.append(self.forward(img, emb_s, att_mask_s))

            outputs = torch.stack(outputs, dim=-1)
            output = outputs.mean(dim=-1)

        # compute I(over N batch) and U(over N batch)
        output = F.interpolate(output, (o_H, o_W), align_corners=True, mode='bilinear')
        pred = output.argmax(1)
        I = torch.sum(torch.mul(pred, target)) * 1.0
        U = torch.sum(torch.add(pred, target)) * 1.0 - I
        IoU = I * 1.0 / U  # [overall IOU of batch]
        self.log('i', I)
        self.log('u', U)
        self.log('iou', IoU)
        return F.cross_entropy(output, target)

        # return {'i': I, 'u': U, 'iou': IoU, 'batch_size': batch_size}

    #def validation_step_end(self, val_step_outputs):
    #    self.log('i', val_step_outputs['i'].item())
    #    self.log('u', val_step_outputs['u'].item())
    #    self.log('iou', val_step_outputs['iou'].item())
        # print('val_step_outputs', val_step_outputs)
        # i, u, iou, bsizes = [], [], [], []
        # for output in val_step_outputs:
        #     print('output', output)
        #     i.append(output['i'])
        #     u.append(output['u'])
        #     iou.append(output['iou'])
        #     bsizes.append(output['batch_size'])
        # i = torch.mean(torch.cat(i))
        # u = torch.mean(torch.cat(u))
        # iou = torch.mean(torch.cat(iou))
        # bsize = sum(bsizes)
        # self.log('i', i)
        # self.log('u', u)
        # self.log('iou', iou)
