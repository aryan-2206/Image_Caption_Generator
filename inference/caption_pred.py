sample_img, sample_caption = dataset[79]
pred_caption, alphas = generate_caption_with_attention(model, sample_img, vocab, device=device)
# print("Actual:", sample_caption)
# print("Predicted:", " ".join(pred_caption))

actual_tokens = [vocab.itos[idx.item()] 
                 for idx in sample_caption 
                 if idx.item() not in [vocab.stoi["<PAD>"], vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<UNK>"]]]

actual_sentence = " ".join(actual_tokens)
print("Actual sentence:", actual_sentence)
print("Predicted sentence:", " ".join(pred_caption))

img_np = sample_img.permute(1, 2, 0).cpu().numpy()
img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
img_np = np.clip(img_np, 0, 1)

plt.imshow(img_np)
plt.axis('off')
plt.title("Input Image")
plt.show()
