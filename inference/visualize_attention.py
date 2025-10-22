sample_img, _ = dataset[79]  
pred_caption, alphas = generate_caption_with_attention(model, sample_img, vocab, device=device)

print("Predicted:", " ".join(pred_caption))
visualize_attention(sample_img, pred_caption, alphas)