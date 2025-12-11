import torch
import torch.nn.functional as F
from collections import defaultdict
from .logit_distillation import logit_distillation_loss


def layer_mapping(teacher_attn_layers, student_attn_layers, mode = "spread"):
    # mode --> spread, one2one-first, one2one-last
    teacher_len, student_len = len(teacher_attn_layers), len(student_attn_layers)
    aligned = defaultdict(set)
    if mode == "one2one-first":
        min_layers = min(teacher_len, student_len)
        for i in range(min_layers):
            aligned[i].add(i)
    elif mode == "one2one-last":
        min_layers = min(teacher_len, student_len)
        for i in range(min_layers-1, -1, -1):
            aligned[teacher_len - min_layers + i].add(student_len - min_layers + i)
    elif mode == "spread":
        if teacher_len <= student_len:
            for student_idx in range(student_len):
                teacher_idx = round(((teacher_len - 1)/(student_len - 1))* student_idx)
                aligned[teacher_idx].add(student_idx)
        else:
            for teacher_idx in range(teacher_len):
                student_idx = round(((student_len - 1)/(teacher_len - 1)) * teacher_idx)
                aligned[teacher_idx].add(student_idx)
    aligned = {k: sorted(v) for k, v in aligned.items()}
    return aligned


def per_attn_layer_loss(teacher_attn, student_attn):
    # based on 	https://doi.org/10.1609/aaai.v38i7.28583 
    batch_size, num_student_heads, seq_len, key_len = student_attn.shape
    batch_size, num_teacher_heads, _, _ = teacher_attn.shape
    student_flatened = student_attn.reshape(batch_size, num_student_heads, -1)
    teacher_flatened = teacher_attn.reshape(batch_size, num_teacher_heads, -1)
    student_normalized = F.normalize(student_flatened, p=1.0, dim=-1)
    teacher_normalized = F.normalize(teacher_flatened, p=1.0, dim=-1)  
    similarity = torch.bmm(teacher_normalized, student_normalized.transpose(1, 2))
    alignment_weights = F.softmax(similarity, dim=-1)
    # weighted_student_attn = torch.bmm(alignment_weights, student_normalized)
    weighted_student_attn = torch.bmm(alignment_weights, student_flatened)
    weighted_student_attn = weighted_student_attn.reshape(batch_size, num_teacher_heads, seq_len, key_len)
    teacher_flattened = teacher_attn.reshape(-1, key_len)
    weighted_student_attn_flatten = weighted_student_attn.reshape(-1, key_len)
    teacher_normalized = F.normalize(teacher_flattened, p=1.0, dim=-1)
    weighted_student_attn_normalized = F.normalize(weighted_student_attn_flatten, p=1.0, dim=-1)
    weighted_student_attn_normalized_log = torch.log(weighted_student_attn_normalized+1e-10)
    attn_loss = F.kl_div(weighted_student_attn_normalized_log, teacher_normalized, reduction="batchmean")
    return attn_loss


def attn_distillation_loss(student_logits, teacher_logits, student_attentions, teacher_attentions, labels, attn_weightage=0.5, ce_weightage = 0.5, include_logit_loss = False, student_loss=None, l_mapping = None, mode = "spread", temperature = 1.0):
    final_loss = 0.0
    if l_mapping is None:
        l_mapping = layer_mapping(teacher_attentions, student_attentions, mode)
    for teacher_idx, student_indices in l_mapping.items():
        teacher_attn = teacher_attentions[teacher_idx]
        pre_concat_student_attn = [student_attentions[i] for i in student_indices]
        concatenated_student_attn = torch.cat(pre_concat_student_attn, dim=1)
        loss = per_attn_layer_loss(teacher_attn, concatenated_student_attn)
        final_loss += loss
    if include_logit_loss:
        logit_loss = logit_distillation_loss(student_logits, teacher_logits, labels, temperature = temperature, ce_weightage = ce_weightage, student_loss=None)
        final_loss = (attn_weightage * final_loss) + ((1 - attn_weightage) * logit_loss)
    return final_loss