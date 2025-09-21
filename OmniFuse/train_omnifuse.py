# train_omnifuse.py
import torch, torch.nn as nn, torch.optim as optim
from config import *
from dataset_multimodal import MultiModalWeldDataset
from encoders import ImageEncoder, SoundEncoder, CurrentEncoder
from cra import CRA
from dwfuse import DWFuse
from tla import TLAMiner
from utils import build_file_list

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # TODO: 先写 file_list，形式为 [(cls, basename), ...]
    # 示例: [("burn_through","burn_through_00001"), ...]
    file_list = build_file_list()
    dataset = MultiModalWeldDataset(IMAGE_ROOT, SOUND_ROOT, CURRENT_ROOT, file_list)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    img_enc = ImageEncoder(512).to(device)
    snd_enc = SoundEncoder(256).to(device)
    cur_enc = CurrentEncoder(in_len=100, out_dim=128).to(device)
    cra = CRA(dim=(512+256+128), K=K).to(device)
    dw = DWFuse([512,256,128], NUM_CLASSES, beta=BETA).to(device)

    params = list(img_enc.parameters()) + list(snd_enc.parameters()) + \
             list(cur_enc.parameters()) + list(cra.parameters()) + list(dw.parameters())
    opt = optim.Adam(params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    tla_miner = TLAMiner(M=3)

    for epoch in range(EPOCHS):
        for imgs, mels, cur, labels in loader:
            imgs, mels, cur, labels = imgs.to(device), mels.to(device), cur.to(device), labels.to(device)

            v_img = img_enc(imgs)
            v_snd = snd_enc(mels)
            v_cur = cur_enc(cur)
            v_concat = torch.cat([v_img, v_snd, v_cur], dim=1)

            v_prime = cra.forward_impute(v_concat)
            v_double = cra.backward_impute(v_prime)

            v_img_p = v_prime[:, :512]
            v_snd_p = v_prime[:, 512:768]
            v_cur_p = v_prime[:, 768:]

            logits_joint, per_logits, omegas = dw([v_img_p, v_snd_p, v_cur_p], labels=labels)

            # Loss
            L_forward = ((v_prime - v_concat)**2).mean()
            L_backward = ((v_double - v_concat)**2).mean()
            per_losses = [criterion(l, labels)*omegas[:,i].mean() for i,l in enumerate(per_logits)]
            L_enc = sum(per_losses)
            L_ra = criterion(logits_joint, labels)
            L_DWFuse = L_enc + ALPHA * L_ra
            L_TLA = 0.0  # TODO: 调用 tla_miner.mine_lazy_mask()

            loss = L_forward + LAMBDA1*L_backward + LAMBDA2*L_DWFuse + LAMBDA3*L_TLA

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss {loss.item():.4f}")

if __name__ == "__main__":
    main()
