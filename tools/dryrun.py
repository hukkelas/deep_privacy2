import click
import torch
import tops
from tops.config import instantiate
from dp2 import utils

@click.command()
@click.argument("config_path")
def run(config_path):
    cfg = utils.load_config(config_path)
    utils.print_config(cfg)

    G = tops.to_cuda(instantiate(cfg.generator))

    D = tops.to_cuda(instantiate(cfg.discriminator))
    cfg.train.batch_size = 2
    print(G)
    dl_val = instantiate(cfg.data.val.loader)
    cfg.train.amp.scaler_D.init_scale = 1
    scaler = instantiate(cfg.train.amp.scaler_D)
    loss_fnc = instantiate(cfg.loss_fnc, D=D, G=G)
    batch = next(iter(dl_val))
    tops.print_module_summary(G, batch, max_nesting=10)
#    tops.print_module_summary(D, batch, max_nesting=10)

    print("G PARAMS:", tops.num_parameters(G) / 10 ** 6)
    print("D PARAMS:", tops.num_parameters(D) / 10 ** 6)
    print(f"Number of trainable parameters in D: {sum(p.numel() for p in D.parameters() if p.requires_grad)/10**6}M")
    print(f"Number of trainable parameters in G: {sum(p.numel() for p in G.parameters() if p.requires_grad)/10**6}M" )

    with torch.cuda.amp.autocast(True):
        o_G = G(**batch)
        o_D = D(**batch)
        print("FORWARD OK")
    D_loss, to_log = loss_fnc.D_loss(batch, grad_scaler=scaler)
    D_loss.backward()
    assert all([p.grad is not None or not p.requires_grad for p in D.parameters()])
    print(to_log)

    G_loss, _ = loss_fnc.G_loss(batch, grad_scaler=scaler)
    G_loss.backward()
    G: torch.nn.Module = G
    for name, p in G.named_parameters():
        if p.grad is None and p.requires_grad:
            print(name)
    assert all([p.grad is not None or not p.requires_grad for p in G.parameters()])

if __name__ == "__main__":
    run()
