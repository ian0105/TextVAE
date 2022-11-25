import argparse
import math
import os
import sys
from pathlib import Path
import time
import pretty_errors
import pytorch_lightning as pl
import torch
import torch.nn as nn
from base_models import datasets, models
from generate import generate
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
from tqdm import tqdm

#from koila import LazyTensor, lazy

def calc_mi(model, dataloader):
    mi = 0
    num_examples = 0
    for batch_data in next(iter(dataloader)):
        batch_size = batch_data.size(0)
        num_examples += batch_size
        mutual_info = calc_mi_q(batch_data)
        mi += mutual_info * batch_size
        print(mi)

    return mi / num_examples
    
pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", help="Config file path")
    parser.add_argument("-d", "--dataset-name", help="Dataset to use")
    parser.add_argument("-t", "--train", action="store_true", help="Run training")
    parser.add_argument(
        "-pe",
        "--pretrain-encoder",
        action="store_true",
        help="Run preliminary encoder training",
    )
    parser.add_argument(
        "-eo",
        "--encoder-only",
        action="store_true",
        help="Exit after encoder pretraining",
    )
    parser.add_argument("-bs", "--batch-size", type=int, help="Batch size", default=1)
    parser.add_argument(
        "-ckpt", "--checkpoint-path", help="Checkpoint path", default=None
    )
    args = parser.parse_args()

    experiment_name = (
        f"{args.dataset_name}_{str(Path(args.config_file).stem)}_bs{args.batch_size}"
    )
    conf = OmegaConf.load(args.config_file)

    base_model = models.get(conf.model_class)
    if not base_model:
        raise Exception("Wrong model.")

    model_class, tokenizer_class = (
        base_model["model_class"],
        base_model["tokenizer_class"],
    )
    tokenizer = tokenizer_class.from_pretrained(conf.base_model_name)

    dataset = datasets.get(args.dataset_name)
    if not dataset:
        raise Exception("Wrong dataset.")

    dataset_class = dataset["dataset_class"]
    out_dim = conf.out_dim
    train_set = dataset_class(dataset["train_file"], tokenizer, out_dim)
    validate_set = (
        dataset_class(dataset["validate_file"], tokenizer, out_dim)
        if dataset["validate_file"]
        else None
    )
    test_set = (
        dataset_class(dataset["test_file"], tokenizer, out_dim)
        if dataset["test_file"]
        else None
    )

    iterations_per_training_epoch = math.ceil(
        dataset["train_dataset_size"] / args.batch_size / torch.cuda.device_count()
    )

    model = model_class(
        tokenizer=tokenizer,
        iterations_per_training_epoch=iterations_per_training_epoch,
        latent_dim=conf.latent_dim,
        pooling_strategy=conf.pooling_strategy,
        min_z=conf.min_z,
        fixed_reg_weight=None,
        denoise_percentage=conf.denoise_percentage,
        base_model=conf.base_model_name,
    )

    cpu_count = os.cpu_count()
    batch_size = args.batch_size
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, num_workers=cpu_count, shuffle = True
    )
    val_dataloader = DataLoader(
        validate_set, batch_size=batch_size, num_workers=cpu_count
    )
    test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=cpu_count)

    pre_mi = 0
    #(train_dataloader, val_dataloader) = lazy(train_dataloader, val_dataloader, batch=0)
    if args.train:

        if args.checkpoint_path:

            model = model_class.load_from_checkpoint(
                args.checkpoint_path,
                strict=False,
                tokenizer=tokenizer,
                iterations_per_training_epoch=iterations_per_training_epoch,
                latent_dim=conf.latent_dim,
                pooling_strategy=conf.pooling_strategy,
                min_z=conf.min_z,
                fixed_reg_weight=None,
                denoise_percentage=conf.denoise_percentage,
                base_model=conf.base_model_name,
            )

            print(f"Loading checkpoint from: {args.checkpoint_path}")

        else:
            start = time.time()
            optimizer = AdamW(model.parameters(),lr=0.001)
            aggressive_flag = True
            report_kl_loss = report_rec_loss = 0
            epoch = 0
            while aggressive_flag:
                print("\nBurning Mode!!\n")
                tk0 = tqdm(train_dataloader,total=len(train_dataloader))
                for i, train_batch in tk0:
                    #encoder
                    train_indices = list(range(len(train_set)))
                    np.random.shuffle(train_indices)
                    train_enc_dataloader = DataLoader(
                    train_set, batch_size=batch_size, num_workers=cpu_count,
                    sampler=SubsetRandomSampler(train_indices[:100*batch_size])
                    )
                    model.freeze_decoder()
                    burn_cur_loss = 0
                    burn_pre_loss = 0
                    for sub_iter, train_enc_batch in enumerate(train_enc_dataloader):
                        optimizer.zero_grad()
                        loss = model.training_step(train_enc_batch[0],sub_iter)
                        burn_cur_loss += loss
                        loss.backward()
                        optimizer.step()
                        if sub_iter % 15 == 0:
                            if burn_pre_loss - burn_cur_loss < 0:
                                break
                            burn_pre_loss = burn_cur_loss
                            burn_cur_loss = 0
                 
                    model.unfreeze_decoder()
                    
                    #decoder
                    model.freeze_encoder()
                    optimizer.zero_grad()
                    loss,recon_loss,reg_loss = model.training_one_step(train_batch,i)
                    loss.backward()
                    optimizer.step()
                    model.unfreeze_encoder()
                    
                    train_loss += loss
                    report_rec_loss += recon_loss
                    report_kl_loss += reg_loss
                    if i % batch_size*10 == 0:
                        model.eval()
                        with torch.no_grad():
                            mi = calc_mi(model, val_dataloader)
                        model.train()
                        print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
                               'time elapsed %.2fs' %
                               (epoch, i, train_loss, report_kl_loss, mi,
                               report_rec_loss, time.time() - start))
                        report_kl_loss = report_rec_loss = 0
                        
                epoch += 1        
                model.eval()
                cur_mi = calc_mi(model, val_dataloader)
                model.train()
                print("pre mi:%.4f. cur mi:%.4f" % (pre_mi, cur_mi))
                if cur_mi - pre_mi < 0:
                    aggressive_flag = False
                    print("\nSTOP BURNING\n")
                pre_mi = cur_mi
                
                    
        # Run regular training.
        early_stop_callback = EarlyStopping(
            # monitor="val_loss",
            monitor="finished_epoch",
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode="min",
            strict=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="finished_epoch",
            mode="max",
            save_weights_only=True,
            save_top_k=10,
        )

        trainer = pl.Trainer(
            gpus=-1,
            accelerator="gpu",
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=10,
            plugins=DDPPlugin(
                find_unused_parameters=True
            ),  # We ignore params from cross-attention.
            log_every_n_steps=1,
            logger=TensorBoardLogger(
                save_dir=os.getcwd(), version=experiment_name, name="lightning_logs"
            ),
            num_sanity_val_steps=0,
        )
        trainer.fit(
            model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    elif args.checkpoint_path:
        print("Test mode")
        model = model_class.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            map_location="cpu",
            tokenizer=tokenizer,
            iterations_per_training_epoch=None,
            latent_dim=conf.latent_dim,
            pooling_strategy=conf.pooling_strategy,
            fixed_reg_weight=None,
            denoise_percentage=conf.denoise_percentage,
            base_model=conf.base_model_name,
        )
        model.eval()
        model.to("cpu")

        fixed_strings = []

        # test_dataloader = DataLoader(test_set, batch_size=args.batch_size)
        test_dataloader = DataLoader(train_set, batch_size=args.batch_size)
        for tokenized, mask, label in test_dataloader:

            # category = category.to(model.master_ctx)
            # tokenized = tokenized.to(model.master_ctx)
            # mask = mask.to(model.master_ctx)

            # model.train()
            # x, z, mu, logvar = model(condition, tokenized, mask, label)
            # loss = x - 1
            # loss.mean().backward()
            # for name, param in model.named_parameters():
            #    if param.grad is None:
            #        print(name)

            # continue
            with torch.no_grad():

                fixed_tokens = generate(
                    model,
                    starter_tokens=[model.config.decoder_start_token_id],
                    input_ids=tokenized,
                    attention_mask=mask,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    top_p=1,
                    top_k=1,
                    temperature=1.0,
                    output_hidden_states=True,
                    num_beams=20,
                    use_cache=True,
                    # sampled_z = torch.ones((1, 64))
                )

            fixed = tokenizer.batch_decode(fixed_tokens, skip_special_tokens=True)
            original = tokenizer.batch_decode(tokenized, skip_special_tokens=True)

            for o, f in zip(original, fixed):

                # print(f"--------\n[CONDITION] {condition}\n[ORIGINAL] {o}\n[FIXED] {f}")
                print(f"--------\n[ORIGINAL] {o}\n[FIXED] {f}")
    else:
        model.eval()
        model.to("cpu")

        fixed_strings = []

        # test_dataloader = DataLoader(test_set, batch_size=args.batch_size)
        test_dataloader = DataLoader(train_set, batch_size=args.batch_size)
        for tokenized, mask, label in test_dataloader:

            # category = category.to(model.master_ctx)
            # tokenized = tokenized.to(model.master_ctx)
            # mask = mask.to(model.master_ctx)

            # model.train()
            # x, z, mu, logvar = model(condition, tokenized, mask, label)
            # loss = x - 1
            # loss.mean().backward()
            # for name, param in model.named_parameters():
            #    if param.grad is None:
            #        print(name)

            # continue
            with torch.no_grad():

                fixed_tokens = generate(
                    model,
                    starter_tokens=[model.config.decoder_start_token_id],
                    input_ids=tokenized,
                    attention_mask=mask,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    top_p=1,
                    top_k=1,
                    temperature=1.0,
                    output_hidden_states=True,
                    num_beams=20,
                    use_cache=True,
                    # sampled_z = torch.ones((1, 64))
                )

            fixed = tokenizer.batch_decode(fixed_tokens, skip_special_tokens=True)
            original = tokenizer.batch_decode(tokenized, skip_special_tokens=True)

            for o, f in zip(original, fixed):

                # print(f"--------\n[CONDITION] {condition}\n[ORIGINAL] {o}\n[FIXED] {f}")
                print(f"--------\n[ORIGINAL] {o}\n[FIXED] {f}")
