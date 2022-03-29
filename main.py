import pytorch_lightning as pl
from torch.utils.data import DataLoader

import config
from args import get_parser
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from model.LAVT import LAVT


def main(args, cfg):
    train_dataset = ReferDataset(args,
                                 split=args.type,
                                 image_transforms=get_transform(args),
                                 eval_mode=args.eval)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              pin_memory=True)
    val_dataset = ReferDataset(args,
                               split='val',
                               image_transforms=get_transform(args),
                               eval_mode=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            pin_memory=True)
    model = LAVT(cfg, args)
    trainer = pl.Trainer(max_epochs=args.epoch,
                         gpus=4,
                         strategy='ddp',
                         gradient_clip_val=0.5,
                         num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=train_loader)  # val_dataloaders=val_loader)


if __name__ == "__main__":
    parse = get_parser()
    args = parse.parse_args()
    cfg = config.get_config(args)
    main(args, cfg)
