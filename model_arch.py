import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MultimodalClassifier(nn.Module):
    def __init__(self, text_model_name, clip_model, num_labels, text_finetune=True, clip_finetune=False):
        super().__init__()
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_hidden = self.text_encoder.config.hidden_size

        # CLIP/SigLIP model (full model passed), we'll use its vision branch + projection
        self.clip_model = clip_model
        
        # Determine image size dynamically
        try:
            image_size = getattr(self.clip_model.config.vision_config, "image_size", 224)
            if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
                h, w = image_size
            else:
                h = w = int(image_size)
        except Exception:
            h = w = 224

        with torch.no_grad():
            dummy = torch.zeros(1, 3, h, w, device=next(self.clip_model.parameters()).device)
            image_emb_sample = self.clip_model.get_image_features(pixel_values=dummy)
        self.image_hidden = int(image_emb_sample.shape[-1])

        # Freeze encoders optionally
        if not text_finetune:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if not clip_finetune:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Fusion MLP - exactly matching V1
        fusion_dim = self.text_hidden + self.image_hidden
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        # Text embeddings (CLS token)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state shape (B, L, H). Use [CLS] at position 0
        text_emb = txt_out.last_hidden_state[:, 0, :]  # (B, H_t)

        # Image embeddings via CLIPModel.get_image_features (gives projected embeddings)
        image_emb = self.clip_model.get_image_features(pixel_values=pixel_values)  # (B, H_i)

        # Concatenate features
        fused = torch.cat([text_emb, image_emb], dim=1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {
            "loss": loss,
            "logits": logits,
            "text_emb": text_emb,
            "image_emb": image_emb
        }


if __name__ == "__main__":
    print("Architecture for MultimodalClassifier")
