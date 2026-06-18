#include <benchmark/benchmark.h>
#include <nlohmann/json.hpp>
#include <semcore/model.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kDatasetPath = "../eval_data/spam_paraphrases.jsonl";

semcore::EmbeddingModelConfig MakeConfig()
{
	semcore::EmbeddingModelConfig config;
	config.onnx_path = "../models/paraphrase-multilingual-MiniLM-L12-v2/model.onnx";
	config.tokenizer_path = "../models/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json";
	config.normalize_embeddings = false;
	config.intra_op_threads = 0;
	config.inter_op_threads = 0;
	config.enable_mem_pattern = true;
	return config;
}

std::vector<std::string> LoadTexts(std::string_view key)
{
	std::ifstream file(std::string{kDatasetPath});
	if (!file.is_open()) {
		throw std::runtime_error("Cannot open dataset");
	}

	std::vector<std::string> texts;
	std::string line;

	while (std::getline(file, line)) {
		auto obj = nlohmann::json::parse(line);
		texts.push_back(obj.at(std::string{key}).get<std::string>());
	}

	return texts;
}

float Dot(std::span<const float> a, std::span<const float> b)
{
	return std::inner_product(a.begin(), a.end(), b.begin(), 0.0F);
}

float CosineSimilarity(std::span<const float> a, std::span<const float> b)
{
	if (a.size() != b.size() || a.empty()) {
		throw std::invalid_argument("Bad vector size");
	}

	const float dot = Dot(a, b);
	const float norm_a = Dot(a, a);
	const float norm_b = Dot(b, b);

	if (norm_a == 0.0F || norm_b == 0.0F) {
		return 0.0F;
	}

	return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

std::span<const float> Row(const semcore::EmbeddingBatch& batch, size_t index)
{
	return batch[index];
}

} // namespace

static void BM_EncodeSingle(benchmark::State& state)
{
	auto texts = LoadTexts("en");
	semcore::EmbeddingModel model(MakeConfig());

	const auto& text = texts.front();

	for (int i = 0; i < 10; ++i) {
		benchmark::DoNotOptimize(model.Encode(text));
	}

	for (auto _ : state) {
		auto embedding = model.Encode(text);
		benchmark::DoNotOptimize(embedding.data());
		benchmark::DoNotOptimize(embedding.size());
	}

	state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_EncodeSingle)->Unit(benchmark::kMillisecond);

static void BM_EncodeBatch(benchmark::State& state)
{
	const auto batch_size = static_cast<size_t>(state.range(0));

	auto texts = LoadTexts("en");
	semcore::EmbeddingModel model(MakeConfig());

	std::vector<std::string> batch;
	batch.reserve(batch_size);

	for (size_t i = 0; i < batch_size; ++i) {
		batch.push_back(texts[i % texts.size()]);
	}

	for (int i = 0; i < 10; ++i) {
		benchmark::DoNotOptimize(model.EncodeBatch(batch));
	}

	for (auto _ : state) {
		auto embeddings = model.EncodeBatch(batch);
		benchmark::DoNotOptimize(embeddings.embeddings.data());
		benchmark::DoNotOptimize(embeddings.embeddings.size());
	}

	state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK(BM_EncodeBatch)
	->Arg(1)
	->Arg(2)
	->Arg(4)
	->Arg(8)
	->Arg(16)
	->Arg(32)
	->Arg(64)
	->Arg(128)
	->Unit(benchmark::kMillisecond);

static void BM_CosineSimilarity(benchmark::State& state)
{
	auto texts = LoadTexts("en");
	semcore::EmbeddingModel model(MakeConfig());
	auto batch = model.EncodeBatch(texts);

	const size_t batch_size = batch.BatchSize();

	for (auto _ : state) {
		for (size_t i = 1; i < batch_size; ++i) {
			const float score = CosineSimilarity(Row(batch, i - 1), Row(batch, i));
			benchmark::DoNotOptimize(score);
		}
	}

	state.SetItemsProcessed(state.iterations() * (batch_size - 1));
}

BENCHMARK(BM_CosineSimilarity)->Unit(benchmark::kNanosecond);

static void BM_DotForNormalizedEmbeddings(benchmark::State& state)
{
	auto texts = LoadTexts("en");
	semcore::EmbeddingModel model(MakeConfig());
	auto batch = model.EncodeBatch(texts);

	const size_t batch_size = batch.BatchSize();

	for (auto _ : state) {
		for (size_t i = 1; i < batch_size; ++i) {
			const float score = Dot(Row(batch, i - 1), Row(batch, i));
			benchmark::DoNotOptimize(score);
		}
	}

	state.SetItemsProcessed(state.iterations() * (batch_size - 1));
}

BENCHMARK(BM_DotForNormalizedEmbeddings)->Unit(benchmark::kNanosecond);

static void BM_TopKLinearScan(benchmark::State& state)
{
	const auto top_k = static_cast<size_t>(state.range(0));

	auto texts = LoadTexts("en");
	semcore::EmbeddingModel model(MakeConfig());
	auto batch = model.EncodeBatch(texts);

	const size_t batch_size = batch.BatchSize();
	const auto anchor = Row(batch, 0);

	std::vector<float> scores(batch_size);

	for (auto _ : state) {
		for (size_t i = 1; i < batch_size; ++i) {
			scores[i] = Dot(anchor, Row(batch, i));
		}

		std::partial_sort(
			scores.begin() + 1,
			scores.begin() + 1 + std::min(top_k, batch_size - 1),
			scores.end(),
			std::greater<>{});

		benchmark::DoNotOptimize(scores.data());
	}

	state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK(BM_TopKLinearScan)
	->Arg(1)
	->Arg(3)
	->Arg(7)
	->Arg(10)
	->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();