#pragma once

// Tiny, dependency-free .env reader for the benchmarks.
//
// It walks up from the current working directory to find the repository root
// (the first ancestor that contains a ".env" or ".git" entry), parses the
// ".env" file there if present, and resolves relative paths against that root.
// This lets a single repo-root .env drive both the Python and the C++
// benchmarks even though the C++ executable runs from core/cmake-build-*/.
//
// Precedence matches python-dotenv with override=False: a real environment
// variable wins over the file value, which wins over the in-code fallback.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace dotenv {

namespace detail {

inline std::string Trim(std::string_view value)
{
	const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
	while (!value.empty() && is_space(value.front())) {
		value.remove_prefix(1);
	}
	while (!value.empty() && is_space(value.back())) {
		value.remove_suffix(1);
	}
	return std::string{value};
}

inline std::string StripQuotes(std::string value)
{
	if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
		value.back() == value.front()) {
		value = value.substr(1, value.size() - 2);
	}
	return value;
}

inline std::string ToLower(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(),
				   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

} // namespace detail

class Env {
   public:
	std::filesystem::path root;

	std::string Get(const std::string& key, const std::string& fallback) const
	{
		if (const char* from_environment = std::getenv(key.c_str())) {
			return from_environment;
		}
		if (auto it = values_.find(key); it != values_.end()) {
			return it->second;
		}
		return fallback;
	}

	int GetInt(const std::string& key, int fallback) const
	{
		const std::string value = Get(key, "");
		if (value.empty()) {
			return fallback;
		}
		try {
			return std::stoi(value);
		} catch (...) {
			return fallback;
		}
	}

	bool GetBool(const std::string& key, bool fallback) const
	{
		const std::string value = detail::ToLower(Get(key, ""));
		if (value.empty()) {
			return fallback;
		}
		if (value == "1" || value == "true" || value == "yes" || value == "on") {
			return true;
		}
		if (value == "0" || value == "false" || value == "no" || value == "off") {
			return false;
		}
		return fallback;
	}

	std::filesystem::path Resolve(const std::string& relative) const
	{
		std::filesystem::path path{relative};
		if (path.is_absolute()) {
			return path;
		}
		return root / path;
	}

	void Set(std::string key, std::string value) { values_[std::move(key)] = std::move(value); }

   private:
	std::unordered_map<std::string, std::string> values_;
};

inline const Env& Environment()
{
	static const Env env = [] {
		namespace fs = std::filesystem;

		Env result;
		fs::path dir = fs::current_path();
		fs::path env_file;
		fs::path repo_root;

		while (true) {
			if (env_file.empty() && fs::exists(dir / ".env")) {
				env_file = dir / ".env";
			}
			if (repo_root.empty() && (fs::exists(dir / ".env") || fs::exists(dir / ".git"))) {
				repo_root = dir;
			}
			if (dir == dir.root_path()) {
				break;
			}
			dir = dir.parent_path();
		}

		result.root = repo_root.empty() ? fs::current_path() : repo_root;

		if (!env_file.empty()) {
			std::ifstream file(env_file);
			std::string line;
			while (std::getline(file, line)) {
				const std::string trimmed = detail::Trim(line);
				if (trimmed.empty() || trimmed.front() == '#') {
					continue;
				}

				std::string body = trimmed;
				if (body.rfind("export ", 0) == 0) {
					body = body.substr(7);
				}

				const auto eq = body.find('=');
				if (eq == std::string::npos) {
					continue;
				}

				std::string key = detail::Trim(body.substr(0, eq));
				std::string value = detail::StripQuotes(detail::Trim(body.substr(eq + 1)));
				if (!key.empty()) {
					result.Set(std::move(key), std::move(value));
				}
			}
		}

		return result;
	}();

	return env;
}

} // namespace dotenv
