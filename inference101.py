import statistics
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Select the best available device.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# Use a longer prompt so the cost of repeated work is easier to observe.
model_id = "openai-community/gpt2"
prompt = "Manchester United is the best club in the world. " * 24
max_new_tokens = 50
benchmark_repeats = 5
# Load the tokenizer and pretrained model.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()
# Convert the prompt into token IDs and move them to the selected device.
inputs = tokenizer(prompt, return_tensors="pt").to(device)
# Generate tokens by processing the full growing sequence during every step.
@torch.inference_mode()
def generate_full_sequence(inputs, max_new_tokens):
    # Copy the prompt tokens and attention mask.
    generated_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"].clone()
    # Generate one token during each iteration.
    for _ in range(max_new_tokens):
        # Process every prompt and generated token again.
        outputs = model(input_ids=generated_ids, attention_mask=attention_mask, use_cache=False)
        # Read the vocabulary scores from the final sequence position.
        next_token_logits = outputs.logits[:, -1, :]
        # Greedy generation selects the token with the highest score.
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        # Add the selected token to the sequence.
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        # Mark the new token as a valid position.
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
        # Stop if the model produces its end-of-sequence token.
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    return generated_ids
# Generate tokens while retaining and reusing the KV cache.
@torch.inference_mode()
def generate_with_cache(inputs, max_new_tokens, return_cache=False):
    # Keep the full sequence for the final decoded output.
    generated_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"].clone()
    # The first call processes the full prompt.
    next_input_ids = generated_ids
    # No K/V tensors exist before the first call.
    past_key_values = None
    # Generate one token during each iteration.
    for _ in range(max_new_tokens):
        # Process the current input and return the updated cache.
        outputs = model(input_ids=next_input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        # Read the vocabulary scores from the final input position.
        next_token_logits = outputs.logits[:, -1, :]
        # Greedy generation selects the token with the highest score.
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        # Add the selected token to the complete output sequence.
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        # Add one valid position to the attention mask.
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
        # Retain the updated K/V tensors.
        past_key_values = outputs.past_key_values
        # Only the newly selected token enters the model next time.
        next_input_ids = next_token_id
        # Stop if the model produces its end-of-sequence token.
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    # Return the cache when it is needed for memory inspection.
    if return_cache:
        return generated_ids, past_key_values
    return generated_ids
# Wait for asynchronous device work to complete.
def synchronize_device():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
# Measure one generation path several times and return the median.
def benchmark(function, inputs, max_new_tokens, repeats):
    times = []
    final_output = None
    for _ in range(repeats):
        # Finish earlier device work before starting the timer.
        synchronize_device()
        start_time = time.perf_counter()
        final_output = function(inputs, max_new_tokens)
        # Wait until all model work has finished.
        synchronize_device()
        times.append(time.perf_counter() - start_time)
    return statistics.median(times), final_output
# Convert current or legacy Hugging Face caches into iterable layer entries.
def get_cache_layers(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values
# Count the bytes occupied by all K and V tensors.
def calculate_kv_cache_bytes(past_key_values):
    total_bytes = 0
    for key_tensor, value_tensor in get_cache_layers(past_key_values):
        total_bytes += key_tensor.numel() * key_tensor.element_size()
        total_bytes += value_tensor.numel() * value_tensor.element_size()
    return total_bytes
# Warm up both execution paths before recording measurements.
generate_full_sequence(inputs, max_new_tokens=2)
generate_with_cache(inputs, max_new_tokens=2)
synchronize_device()
# Measure total generation time for both paths.
full_sequence_time, full_sequence_ids = benchmark(generate_full_sequence, inputs, max_new_tokens, benchmark_repeats)
cached_time, cached_ids = benchmark(generate_with_cache, inputs, max_new_tokens, benchmark_repeats)
# Both greedy paths should produce the same token IDs.
assert torch.equal(full_sequence_ids, cached_ids), "The two generation paths produced different token IDs."
# Run the cached path once more and retain its final KV cache.
cached_ids_for_memory, past_key_values = generate_with_cache(inputs, max_new_tokens, return_cache=True)
# Calculate timing and throughput values.
prompt_tokens = inputs["input_ids"].shape[1]
generated_tokens = cached_ids.shape[1] - prompt_tokens
full_sequence_throughput = generated_tokens / full_sequence_time
cached_throughput = generated_tokens / cached_time
speedup = full_sequence_time / cached_time
time_reduction_percent = (full_sequence_time - cached_time) / full_sequence_time * 100
# Calculate the size of the actual K/V tensors.
cache_layers = get_cache_layers(past_key_values)
kv_cache_bytes = calculate_kv_cache_bytes(past_key_values)
first_key_tensor, first_value_tensor = cache_layers[0]
cached_sequence_length = first_key_tensor.shape[-2]
kv_bytes_per_cached_token = kv_cache_bytes / cached_sequence_length
# Decode the generated token IDs for inspection.
output_text = tokenizer.decode(cached_ids[0], skip_special_tokens=True)
# Display the benchmark configuration.
print(f"Model: {model_id}")
print(f"Device: {device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Prompt tokens: {prompt_tokens}")
print(f"Generated tokens: {generated_tokens}")
print(f"Benchmark repetitions: {benchmark_repeats}")
print()
# Display timing and throughput results.
print(f"Full-sequence generation: {full_sequence_time:.3f} seconds")
print(f"KV-cache generation: {cached_time:.3f} seconds")
print(f"Full-sequence throughput: {full_sequence_throughput:.2f} tokens/second")
print(f"KV-cache throughput: {cached_throughput:.2f} tokens/second")
print(f"KV-cache speedup: {speedup:.2f}x")
print(f"Generation-time reduction: {time_reduction_percent:.1f}%")
print()
# Display KV-cache shape and memory results.
print(f"KV-cache layers: {len(cache_layers)}")
print(f"First-layer K shape: {tuple(first_key_tensor.shape)}")
print(f"First-layer V shape: {tuple(first_value_tensor.shape)}")
print(f"Cached sequence length: {cached_sequence_length}")
print(f"KV-cache size: {kv_cache_bytes:,} bytes")
print(f"KV-cache size: {kv_cache_bytes / (1024**2):.2f} MiB")
print(f"KV-cache bytes per cached token: {kv_bytes_per_cached_token:,.0f}")
print()
# Display the final generated text.
print("Generated text:")
print(output_text)
