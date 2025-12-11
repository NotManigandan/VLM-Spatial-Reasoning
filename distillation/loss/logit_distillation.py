import torch
import torch.nn.functional as F

def logit_distillation_loss(student_logits, teacher_logits, labels, temperature = 1.0, ce_weightage = 0.5, student_loss=None):
    # based on https://arxiv.org/abs/1503.02531
    student_logits, teacher_logits = student_logits[:, :-1, :], teacher_logits[:, :-1, :]
    labels = labels[:, 1:]
    not_ignore = (labels != -100)
    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    ce_loss = ce_criterion(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1))
    student_logits, teacher_logits = student_logits[not_ignore]/temperature, teacher_logits[not_ignore]/temperature
    student_prob = F.log_softmax(student_logits, dim=-1)
    teacher_prob = F.log_softmax(teacher_logits, dim=-1)
    kl = F.kl_div(student_prob, teacher_prob, reduction="batchmean", log_target=True) 
    final_loss = ce_weightage * ce_loss + ((1 - ce_weightage) * (temperature ** 2) * kl)
    return final_loss