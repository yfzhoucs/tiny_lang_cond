import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms as T
from tqdm.auto import tqdm
from transformers import BertTokenizerFast as BertTokenizer, BertModel
from torch.autograd import Variable


class RobotExperiments(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 tokenizer: BertTokenizer,
                 transforms=None
                 ):
        self.root = root
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.max_token_len = 512
        self.experiments = []
        self.scaler = MinMaxScaler()
        for item in list(sorted(os.listdir(root))):
            exp_dir = os.path.join(root, item)
            if os.path.isdir(exp_dir) and 'user_inputs.json' in os.listdir(exp_dir):
                joints_file = os.path.join(exp_dir, 'joints.npy')
                joints = np.load(joints_file)

                end_position_file = os.path.join(exp_dir, 'end_positions.npy')
                end_positions = np.load(end_position_file)

                joints_and_end_positions = np.concatenate((end_positions, joints), axis=1)
                self.scaler.partial_fit(joints_and_end_positions)
                self.experiments.append(exp_dir)
                self.experiments.append(exp_dir)

    def __getitem__(self, idx):
        # load images, joints, user inputs and end positions of robot arm
        start_img_path = os.path.join(self.root, self.experiments[idx], '0.png')
        images = [item for item in os.listdir(os.path.join(self.root, self.experiments[idx])) if '.png' in item.lower()]
        images.sort(key=lambda x: int(x[:-4]))
        first_image = Image.open(start_img_path).convert("RGB")
        last_image = Image.open(os.path.join(self.root, self.experiments[idx], images[-1])).convert("RGB")

        user_input_file, experiment_user_input_match = None, None
        if idx % 2 == 0:
            experiment_user_input_match = torch.tensor(1, dtype=torch.float)
            user_input_file = os.path.join(self.root, self.experiments[idx], 'user_inputs.json')
        else:
            experiment_user_input_match = torch.tensor(0, dtype=torch.float)
            user_input_file = os.path.join(self.root, self.experiments[idx], 'wrong_user_inputs.json')
        joints_file = os.path.join(self.root, self.experiments[idx], 'joints.npy')
        end_position_file = os.path.join(self.root, self.experiments[idx], 'end_positions.npy')

        user_input, joints, end_positions = None, None, None
        if os.path.isfile(user_input_file):
            with open(user_input_file, 'r') as f:
                data = json.load(f)
                joints = np.load(joints_file)
                end_positions = np.load(end_position_file)
                joints_and_end_positions = np.concatenate((end_positions, joints), axis=1)
                for username in data.keys():
                    random_int = np.random.randint(0, 1)
                    user_input = data[username][random_int]
                    break

        user_input_encoded = self.tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        if self.transforms is not None:
            first_image = self.transforms(first_image)
            last_image = self.transforms(last_image)

        joints_and_end_positions = torch.tensor(joints_and_end_positions, dtype=torch.float)
        return first_image, last_image, user_input_encoded, joints_and_end_positions, experiment_user_input_match

    def __len__(self):
        return len(self.experiments)


class BERTMLESentiment(nn.Module):
    def __init__(self, bert, dropout, lstm_input_dim, lstm_hidden_dim, output_dim, device, num_layers=4):
        super().__init__()
        # text embedding
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, 512)
        self.dropout = nn.Dropout(dropout)

        # First and Last image embeddings
        cnn_model = torchvision.models.resnet18(pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(cnn_model.children())[:-1]), nn.Flatten())

        # joints and end-positions embedding
        self.hidden_layer_size = lstm_hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(lstm_input_dim, self.hidden_layer_size, num_layers=self.num_layers, batch_first=True)
        self.lstm_linear = nn.Linear(self.hidden_layer_size*self.num_layers, output_dim)

        self.multihead_attn = nn.MultiheadAttention(512, 4)
        self.final_fc = nn.Linear(4*512, 1)
        self.device = device

    def forward(self, input_ids, attention_mask, first_image_batch, last_image_batch, joints_and_endpos_batch):
        # Text batch : (batch_size, variable_length)
        text_embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]  # (batch_size, 768)
        hidden = self.dropout(text_embedding)  # (batch_size, 768)
        hidden = F.relu(self.fc(hidden))  # (batch_size, 512)
        text_out = self.dropout(hidden)  # (batch_size, 512)

        # first image batch : (batch_size, 3, 128, 128)
        first_batch_out = self.backbone(first_image_batch)  # (batch_size, 512)
        # last image batch : (batch_size, 3, 128, 128)
        last_batch_out = self.backbone(last_image_batch)  # (batch_size, 512)

        # joints and end positions: (batch_size, variable_length, 4)
        # (batch_size, variable_length, 100)
        h_0 = Variable(torch.zeros(self.num_layers, 1, self.hidden_layer_size)).to(self.device)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, 1, self.hidden_layer_size)).to(self.device)  # internal state
        lstm_hidden, (hn, cn) = self.lstm(joints_and_endpos_batch, (h_0, c_0))
        hn = hn.view(-1, self.hidden_layer_size*self.num_layers)
        lstm_out = self.lstm_linear(hn)  # (batch_size, 512)

        stacked_output = torch.stack((text_out, first_batch_out, last_batch_out, lstm_out), dim=1)
        # stacked_lstm_output = torch.stack((lstm_out, lstm_out, lstm_out), dim=1)
        attn_output, attn_output_weights = self.multihead_attn(stacked_output, stacked_output, stacked_output)
        attn_output = torch.flatten(attn_output, start_dim=1)
        attn_output = torch.sigmoid(self.final_fc(attn_output))
        return attn_output


def get_transform():
    transforms = [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # converts the image, a PIL image, into a PyTorch Tensor
    return T.Compose(transforms)


def prepare_dataloaders(data_dir, tokenizer):
    dataset = RobotExperiments(data_dir, tokenizer, transforms=get_transform())
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-11])
    dataset_test = torch.utils.data.Subset(dataset, indices[-11:])
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=1)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1)
    return data_loader_train, data_loader_test


def binary_accuracy(preds, y):
    """ Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8 """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train_one_epoch(final_model, optimizer, iterator, loss_criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    final_model.train()
    for batch in tqdm(iterator):
        first_image, last_image, user_input_encoded, joints_and_end_positions, label = batch
        first_image, last_image = first_image.to(device), last_image.to(device)
        joints_and_end_positions = joints_and_end_positions.to(device)
        input_ids = user_input_encoded.get('input_ids').squeeze(0)
        attention_mask = user_input_encoded.get('attention_mask')
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predictions = final_model(input_ids, attention_mask, first_image, last_image, joints_and_end_positions).squeeze(1)
        loss = loss_criterion(predictions, label)
        acc = binary_accuracy(predictions, label)
        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # txt = "Training_loss: {loss:.2f}\tAccuracy: {accuracy:.2f}\t Iteration: {Iteration}"
        # print(txt.format(loss=epoch_loss/count, accuracy=epoch_acc/count, Iteration=count))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(final_model, iterator, loss_criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    final_model.eval()

    with torch.no_grad():
        for batch in tqdm(iterator):
            first_image, last_image, user_input_encoded, joints_and_end_positions, label = batch
            first_image, last_image = first_image.to(device), last_image.to(device)
            joints_and_end_positions = joints_and_end_positions.to(device)
            input_ids = user_input_encoded.get('input_ids').squeeze(0)
            attention_mask = user_input_encoded.get('attention_mask')
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            label = label.to(device)

            predictions = final_model(user_input_encoded, first_image, last_image, joints_and_end_positions).squeeze(1)
            loss = loss_criterion(predictions, label)
            acc = binary_accuracy(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def fit(num_epochs,
        final_model,
        optimizer,
        data_loader_train,
        data_loader_test,
        checkpoint_dir,
        loss_criterion,
        learning_rate_scheduler,
        device
        ):
    torch.autograd.set_detect_anomaly(True)
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_loss, train_acc = train_one_epoch(final_model, optimizer, data_loader_train, loss_criterion, device)
        # update the learning rate
        learning_rate_scheduler.step()
        # evaluate on the test dataset
        valid_loss, valid_acc = evaluate(final_model, data_loader_test, loss_criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(final_model.state_dict(), os.path.join(checkpoint_dir, 'resnet50-23-APR-table.pth'))

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


if __name__ == '__main__':
    BERT_MODEL_NAME = 'bert-base-cased'
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    root_dir = r'C:\Users\arpit\PycharmProjects\tiny_lang_cond\data_position_random_shape_30_25_part1'
    train_data_loader, test_data_loader = prepare_dataloaders(root_dir, bert_tokenizer)
    compute_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset = RobotExperiments(root_dir, bert_tokenizer, transforms=get_transform())
    # start_img, end_img, user_input_encoding, joints_and_end_pos, label = dataset[3]

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    num_classes = 2
    model = BERTMLESentiment(bert_model, dropout=0.3, lstm_input_dim=4, lstm_hidden_dim=100,
                             output_dim=512, device=compute_device)
    model.to(compute_device)
    params = [p for p in model.parameters() if p.requires_grad]
    adam_optimizer = torch.optim.Adam(
        params, lr=0.001,
        betas=(0.9, 0.997),
        weight_decay=0.0001
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        adam_optimizer,
        step_size=15,
        gamma=0.5
    )
    criterion = nn.BCEWithLogitsLoss()
    fit(10, model, adam_optimizer, train_data_loader, test_data_loader, root_dir, criterion,
        lr_scheduler, compute_device)

    # start_img = start_img[None, :]
    # end_img = end_img[None, :]
    # joints_and_end_pos = joints_and_end_pos[None, :]
    # joints_and_end_pos = torch.FloatTensor(joints_and_end_pos)
    # score = model(user_input_encoding, start_img, end_img, joints_and_end_pos)
    # print(score)
