import torch
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import inspect
from tqdm import tqdm


class Trainer():
    def __init__(self, model, optims, samplers, crit, device,
                 reg=None, callbacks=None, iterations=None):
        self.model = model
        self.optims = optims
        self.samplers = samplers
        self.device = device
        self.crit = crit
        self.reg = reg
        self.callbacks = callbacks
        self.gen_iters, self.dist_iters = iterations \
            if iterations is not None else (1, 1)
        if self.reg is not None:
            self.reg = [(r, inspect.getfullargspec(
                r.forward)[0][1:]) for r in self.reg]
        self.description, self.pbar_description = \
            self.get_description()

    def train(self, dataloader, epochs, output_dir=None,
              output_transform=None, clear_dir=False):
        if clear_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=False)
        self.history = []
        for e in range(1, epochs+1):
            self.epoch = e
            epoch_history, sample_data = self.train_epoch(dataloader)
            self.history.extend(epoch_history)
            if self.callbacks is not None and 'epoch' in self.callbacks:
                for cb in self.callbacks['epoch']:
                    cb()
            if output_dir is not None:
                self.save_results(sample_data, output_dir, output_transform)
        self.history = pd.DataFrame(self.history, columns=self.description)
        if output_dir is not None:
            self.save_history(output_dir)
        return self.history

    def train_epoch(self, dataloader):
        for net in self.model.values():
            net.train()
        history = []
        pbar = tqdm(dataloader)
        for real_samples in pbar:
            losses = {}
            data = self.sample_data(real_samples)
            losses['disc_main'], losses['disc_real'], \
                losses['disc_fake'], losses['disc_reg'] = \
                self.train_discriminator(data)
            losses['gen_main'], losses['gen_partial'] = \
                self.train_generator(data)
            if self.callbacks is not None and 'iteration' in self.callbacks:
                for cb in self.callbacks['iteration']:
                    cb()
            history.append(self.get_iteration_history(losses))
            pbar.set_description(self.pbar_description.format(
                self.epoch, *history[-1]))
        return history, data

    def sample_data(self, real_samples):
        data = {}
        batch_size = real_samples.shape[0]
        real_labels, fake_labels = self.samplers['labels'](batch_size)
        data['real_labels'], data['fake_labels'] = real_labels.to(
            self.device), fake_labels.to(self.device)
        data['latent'] = self.samplers['latent'](batch_size).to(self.device)
        data['real_samples'] = real_samples.to(self.device)
        return data

    def train_discriminator(self, data):
        for opt in self.optims.values():
            opt.zero_grad()
        fake_samples = self.model['gen'](data['latent']).detach()
        fake_preds = self.model['disc'](fake_samples)
        real_preds = self.model['disc'](data['real_samples'])
        fake_losses = [c(fake_preds, data['fake_labels']) for c in self.crit]
        real_losses = [c(real_preds, data['real_labels']) for c in self.crit]
        main_loss = torch.sum(torch.stack([*fake_losses, *real_losses]))
        if self.reg is not None:
            data['fake_samples'] = fake_samples
            reg_loss = self.regularize_discriminator(data)
            main_loss += torch.sum(torch.stack(reg_loss))
        else:
            reg_loss = None
        main_loss.backward()
        self.optims['disc'].step()
        return main_loss, real_losses, fake_losses, reg_loss

    def regularize_discriminator(self, data):
        reg_kwargs = {**self.model, **data}
        return [r[0].forward(*list(map(reg_kwargs.get, r[1])))
                for r in self.reg]

    def train_generator(self, data):
        for opt in self.optims.values():
            opt.zero_grad()
        fake_samples = self.model['gen'](data['latent'])
        fake_preds = self.model['disc'](fake_samples)
        gen_partial = [c(fake_preds, data['real_labels']) for c in self.crit]
        main_loss = torch.sum(torch.stack(gen_partial))
        main_loss.backward()
        self.optims['gen'].step()
        return main_loss, gen_partial

    def get_iteration_history(self, losses):
        iteration = [losses['gen_main'].item(), losses['disc_main'].item()]
        iteration.extend([loss.item() for loss in itertools.chain(
            losses['disc_real'], losses['disc_fake'])])
        if losses['disc_reg'] is not None:
            iteration.extend([loss.item() for loss in losses['disc_reg']])
        iteration.extend([loss.item() for loss in losses['gen_partial']])
        return iteration

    def get_description(self):
        description = [
            'gen', 'disc',
            *[f"disc_real({c.__class__.__name__})" for c in self.crit],
            *[f"disc_fake({c.__class__.__name__})" for c in self.crit]
        ]
        if self.reg is not None:
            description.extend(
                [f"disc_reg({r[0].__class__.__name__})" for r in self.reg])
        description.extend(
            [f"gen_partial({c.__class__.__name__})" for c in self.crit])
        pbar_description = "epoch: {0}; " \
            + "; ".join([f"{d}: {{{i+1}:10.6f}}"
                         for i, d in enumerate(description)])
        return description, pbar_description

    def save_results(self, data, output_dir, output_transfrom):
        for net in self.model.values():
            net.eval()
        epoch_output_dir = os.path.join(output_dir, f'epoch_{self.epoch}')
        os.mkdir(epoch_output_dir)
        with torch.no_grad():
            fake_images = self.model['gen'](data['latent']).cpu()
        for i, image in enumerate(torch.split(fake_images, 1)):
            image = output_transfrom(image)
            image.save(os.path.join(
                epoch_output_dir, f'e{self.epoch}_{i}.png'))
        torch.save(self.model['gen'].state_dict(), os.path.join(
            epoch_output_dir, f'{self.epoch}_gen.pth'))
        torch.save(self.model['disc'].state_dict(), os.path.join(
            epoch_output_dir, f'{self.epoch}_disc.pth'))

    def save_history(self, output_dir):
        self.history.to_pickle(os.path.join(output_dir, 'train_history.pkl'))
        fig = plt.figure(figsize=(24, 12))
        self.history.plot(alpha=0.2, ax=fig.gca())
        self.history.rolling(32, center=True).mean().plot(ax=fig.gca())
        fig.savefig(os.path.join(output_dir, 'train_history.png'),
                    transparent=False, edgecolor='w',
                    dpi=200, bbox_inches='tight')
