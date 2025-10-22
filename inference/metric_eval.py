# Evaluate BLEU
subset_indices = list(range(1000))  # evaluate on first 500 images
val_loader = DataLoader(Subset(dataset, subset_indices), batch_size=32, shuffle=False, collate_fn=collate_fn)
bleu1, bleu4 = evaluate_bleu(model, val_loader, vocab, device)

# bleu1, bleu4 = evaluate_bleu(model, dataloader, vocab, device)
print(f"BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}")

# visualize_attention(sample_img, pred_caption, alphas)