import socket
from argparse import ArgumentParser
import os

from torch.nn import Softmax
from pytorch_lightning import seed_everything

from mobile_data import C10IMGDATA_MOBILE
from utils.all_classifiers import all_classifiers

HOST = '192.168.6.154'  # change it before usage!
PORT = 8081
DATA_DIR = "/home/tangbao/codes/ProtectivePerturbation_MMSys22/recv_from_phone"  # change it before usage!


def get_args():
    parser = ArgumentParser()
    seed_everything(0)
    parser.add_argument("--description", type=str, default="protective perturbation mmsys 2022")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu_id", type=str, default="0")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    m = Softmax(dim=1)
    client_dataset = C10IMGDATA_MOBILE(args).dataloader()
    client_dataset = iter(client_dataset)
    target_model = all_classifiers["mobilenet_v2"](pretrained=True).eval().cuda()

    skt = socket.socket()
    try:
        skt.bind((HOST, PORT))
        skt.listen(5)
        print("Init server successfully.")
        print("Waiting for connection...")
    except:
        print("Server initialization error!")
        exit(1)

    cnt = 0
    while True:
        conn, addr = skt.accept()
        print("New connection", conn, addr)

        # receive img size first
        img_size = int(str(conn.recv(16), encoding='utf-8').strip())
        if img_size != 0:
            print("Receiving image w/ size", img_size)
            img_path = 'recv_from_phone/' + str(cnt) + '.png'
            cnt += 1
            f = open(img_path, 'wb')
        else:
            continue

        recv_size = 0
        while True:
            if recv_size == img_size:
                break
            left_size = img_size - recv_size
            if left_size >= 1024:
                data = conn.recv(1024)
            else:
                data = conn.recv(left_size)
            f.write(data)
            accu_size = len(data)
            recv_size = recv_size + accu_size
        f.close()
        print('Received', img_path, 'successfully')

        client_data = next(client_dataset)
        client_data = client_data.cuda()
        client_data_predict = target_model(client_data)
        client_data_predict = m(client_data_predict)

        label = client_data_predict.argmax().item()
        label_str = str(label)
        print('predicted label', label_str)
        conn.sendall(bytes(label_str, encoding='utf-8'))
        conn.close()
    skt.close()
