import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from spel.configuration import device, get_base_model_name
from cgen.data_loader import get_dataset, spel_annotate, tokenizer, SpEL_instance, wikipedia2vec_instance

class Transformation(nn.Module):
    def __init__(self, spel_lm_size, w2v_size=768, hidden_size=256):
        super(Transformation, self).__init__()
        self.w2v_map = nn.Linear(w2v_size, hidden_size)
        self.spel_map = nn.Linear(spel_lm_size, hidden_size)
        nn.init.xavier_uniform_(self.w2v_map.weight)
        nn.init.xavier_uniform_(self.spel_map.weight)


class SpELTransformation(nn.Module):
    def __init__(self, spel_lm_size, w2v_size, hidden_size=256):
        super(SpELTransformation, self).__init__()
        self.fc1 = nn.Linear(spel_lm_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, w2v_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Generator:
    def __init__(self, only_transform_spel=True):
        self.only_transform_spel = only_transform_spel
        transform_class = SpELTransformation if only_transform_spel else Transformation
        self.transformation = transform_class(1024 if get_base_model_name() == 'roberta-large' else 768, 768).to(device)

    def train(self, train_on_aida=True, lm_batch_size=1, triple_vector_batch_size=64, lr=5e-4, num_epochs=1, checkpoint_every=1000,
              save_model_name='generator_w.pt', expected_negative_examples=10, default_key = 'start'):
        optimizer = optim.Adam(self.transformation.parameters(), lr=lr)
        criterion = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        total_loss = 0
        cnt_loss = 0
        for _ in range(num_epochs):
            # Run the training code to train self.transformation
            dataset = tqdm(get_dataset(train_on_aida, triple_vector_batch_size, expected_negative_examples, default_key = default_key, lm_batch_size=lm_batch_size))
            for batch_id, batch in enumerate(dataset):
                if batch_id > 0 and batch_id % checkpoint_every == 0:
                    print('checkpointing ...')
                    torch.save(self.transformation.state_dict(), save_model_name)
                anchor, positive, negative = batch
                if self.only_transform_spel:
                    anchor = self.transformation(anchor)
                else:
                    anchor = self.transformation.spel_map(anchor)
                    positive = self.transformation.w2v_map(positive)
                    negative = self.transformation.w2v_map(negative)
                loss = criterion(anchor, positive, negative)
                del batch[:]
                total_loss += loss.detach().item()
                cnt_loss += 1.0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                dataset.set_description(f"Avg Loss: {total_loss/cnt_loss:.7f}")
            torch.save(self.transformation.state_dict(), save_model_name)


if __name__ == "__main__":
    cgen = Generator()
    cgen.train(train_on_aida=True, lm_batch_size=10, triple_vector_batch_size=64, lr=5e-4, num_epochs=50, checkpoint_every=2000,
              save_model_name='generator_w.pt', expected_negative_examples=10, default_key = 'start')
