import typing

from datasets import load_dataset
from transformers import AutoImageProcessor
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import AutoModelForImageClassification, MobileNetV2Config, MobileNetV2ForImageClassification
import evaluate
import numpy as np
from transformers import DefaultDataCollator


class ImageDistilTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, lambda_param=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, student, inputs, return_outputs=False):
        student_output = self.student(**inputs)

        with torch.no_grad():
            inputs['labels'] = inputs['labels'].to(torch.device("cpu"))
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.device("cpu"))
            teacher_output = self.teacher(**inputs)

        # Compute soft targets for teacher and student
        soft_teacher = f.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = f.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the loss
        soft_student = soft_student.to(torch.device("cpu"))
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # Compute the true label loss
        student_target_loss = student_output.loss

        # Calculate final loss
        loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss


def message(d: typing.Any):
    return {
        "event": "message",
        "retry": 15000,
        "data": d
    }


def trainAndEvaluate(
        num_train_epochs, fp16, logging_strategy, evaluation_strategy, save_strategy, load_best_model_at_end,
        temperature, lambda_param,
):
    yield message("timeline:Processing has started")

    dataset = load_dataset("beans")

    yield message("timeline:Dataset loaded")

    teacher_processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

    def process(examples):
        return teacher_processor(examples["image"])

    processed_datasets = dataset.map(process, batched=True)

    yield message("timeline:Apply the preprocessing to split of the dataset")

    # initialize models
    num_labels = len(processed_datasets["train"].features["labels"].names)
    teacher_model = AutoModelForImageClassification.from_pretrained(
        "merve/beans-vit-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    yield message("timeline:Teacher model initialized")

    # training MobileNetV2 from scratch
    student_config = MobileNetV2Config()
    student_config.num_labels = num_labels
    student_model = MobileNetV2ForImageClassification(student_config)

    yield message("timeline:Student model initialized")

    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=num_train_epochs,
        fp16=fp16,
        logging_dir="logs",
        logging_strategy=logging_strategy,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="accuracy",
        report_to=["tensorboard"],
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id="demo",
    )

    yield message("timeline:Training Arguments have been set")

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
        return {"accuracy": acc["accuracy"]}

    data_collator = DefaultDataCollator()
    trainer = ImageDistilTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        temperature=temperature,
        lambda_param=lambda_param,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        data_collator=data_collator,
        tokenizer=teacher_processor,
        compute_metrics=compute_metrics,
    )

    yield message("timeline:Trainer initialized")

    yield message("timeline:Training starts")

    result = trainer.train()

    yield message("timeline:Training completed")

    yield message(f"result:{result}")

    yield message("timeline:Evaluation starts")

    evaluation = trainer.evaluate(processed_datasets["test"])

    yield message("timeline:Evaluation completed")

    yield message(f"evaluation:{evaluation}")
