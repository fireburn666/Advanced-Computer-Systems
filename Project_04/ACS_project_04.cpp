#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <queue>
#include <functional>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <immintrin.h> // For SIMD
using namespace std;

// Timer
class Timer {
    using Clock = chrono::high_resolution_clock;
    Clock::time_point start_time;
public:
    void start() { start_time = Clock::now(); }
    double elapsed() {
        auto end_time = Clock::now();
        return chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    }
};

// Data Structures 
struct Dictionary {
    unordered_map<string, int> data_to_id;
    vector<string> id_to_data;
};
struct Encoded_Column {
    Dictionary dictionary;
    vector<int> encoded_data;
};

// File Handling
vector<string> read_column_from_file(const string& filename) {
    ifstream file(filename);
    vector<string> data;
    string line;
    while (getline(file, line)) {
        data.push_back(line);
    }
    return data;
}

void encoded_column_to_file(const Encoded_Column& encoded_column, const string& filename) {
    ofstream file(filename);
    if (!file) {
        cerr << "Error: Unable to open file for writing: " << filename << "\n";
        return;
    }
    // Write the dictionary to the file
    file << "Dictionary:\n";
    for (size_t i = 0; i < encoded_column.dictionary.id_to_data.size(); ++i) {
        file << i << ": " << encoded_column.dictionary.id_to_data[i] << "\n";
    }
    // Write the encoded data
    file << "\nEncoded Data:\n";
    for (int id : encoded_column.encoded_data) {
        file << id << " ";
    }
    file << "\n";
    file.close();
}

// Thread worker for encoding chunks
void encode_chunk(const vector<string>& input_data, size_t start, size_t end, unordered_map<string, int>& dict, vector<string>& id_to_data, vector<int>& encodedChunk) {
    for (size_t i = start; i < end; ++i) {
        const auto& item = input_data[i];
        if (dict.find(item) == dict.end()) {
            int id = static_cast<int>(id_to_data.size());
            dict[item] = id;
            id_to_data.push_back(item);
        }
        encodedChunk.push_back(dict[item]);
    }
}

// Merge local dictionaries into the global dictionary
void merge_dict(const unordered_map<string, int>& dict, const vector<string>& id_to_data, Dictionary& main_dict, unordered_map<int, int>& local_to_global_map, mutex& global_mutex) {
    for (size_t local_id = 0; local_id < id_to_data.size(); ++local_id) {
        const string& item = id_to_data[local_id];
        lock_guard<mutex> lock(global_mutex);
        if (main_dict.data_to_id.find(item) == main_dict.data_to_id.end()) {
            int global_id = static_cast<int>(main_dict.id_to_data.size());
            main_dict.data_to_id[item] = global_id;
            main_dict.id_to_data.push_back(item);
        }
        local_to_global_map[local_id] = main_dict.data_to_id[item];
    }
}

// Perform dictionary encoding with multithreading
Encoded_Column encode_dict(const vector<string>& input_data, int num_threads) {
    size_t data_size = input_data.size();
    Encoded_Column encoded_column;
    encoded_column.encoded_data.resize(data_size);
    size_t chunk_size = (data_size + num_threads - 1) / num_threads;
    vector<unordered_map<string, int>> dicts(num_threads);
    vector<vector<string>> id_to_data(num_threads);
    vector<vector<int>> encoded_chunks(num_threads);
    vector<thread> threads;
    mutex global_mutex;
    vector<unordered_map<int, int>> local_to_global_maps(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = min(start + chunk_size, data_size);

        if (start < end) {
            threads.emplace_back(
                encode_chunk,
                cref(input_data),
                start,
                end,
                ref(dicts[t]),
                ref(id_to_data[t]),
                ref(encoded_chunks[t]));
        }
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (int t = 0; t < num_threads; ++t) {
        merge_dict(dicts[t], id_to_data[t], encoded_column.dictionary, local_to_global_maps[t], global_mutex);
    }
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        for (size_t i = 0; i < encoded_chunks[t].size(); ++i) {
            encoded_column.encoded_data[start + i] = local_to_global_maps[t][encoded_chunks[t][i]];
        }
    }
    return encoded_column;
}

// SIMD prefix query search
vector<pair<string, vector<int>>> prefix_query_simd(const Encoded_Column& encoded_column, const string& prefix) {
    vector<pair<string, vector<int>>> results;
    size_t prefix_length = prefix.size();
    if (prefix_length == 0) {
        cerr << "Error: Prefix length cannot be zero.\n";
        return results;
    }
    const auto& dict_entries = encoded_column.dictionary.id_to_data;
    int prefix_mask = (1 << prefix_length) - 1;
    __m256i prefix_vec = _mm256_setzero_si256();
    memcpy(&prefix_vec, prefix.c_str(), min(prefix_length, size_t(32)));
    size_t i = 0;
    for (; i + 8 <= dict_entries.size(); i += 8) {
        alignas(32) char dict_buffer[8][32] = {0}; 
        for (int j = 0; j < 8; ++j) {
            const auto& dict_entry = dict_entries[i + j];
            memcpy(dict_buffer[j], dict_entry.c_str(), min(dict_entry.size(), size_t(32)));
        }
        for (int j = 0; j < 8; ++j) {
            __m256i dict_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dict_buffer[j]));
            __m256i cmp = _mm256_cmpeq_epi8(prefix_vec, dict_vec);
            int mask = _mm256_movemask_epi8(cmp);
            if ((mask & prefix_mask) == prefix_mask) {
                const string& dict_entry = dict_entries[i + j];
                vector<int> indices;
                for (size_t k = 0; k < encoded_column.encoded_data.size(); ++k) {
                    if (encoded_column.encoded_data[k] == static_cast<int>(i + j)) {
                        indices.push_back(k);
                    }
                }
                results.emplace_back(dict_entry, indices);
            }
        }
    }
    for (; i < dict_entries.size(); ++i) {
        const auto& dict_entry = dict_entries[i];
        if (dict_entry.size() >= prefix_length && dict_entry.compare(0, prefix_length, prefix) == 0) {
            vector<int> indices;
            for (size_t k = 0; k < encoded_column.encoded_data.size(); ++k) {
                if (encoded_column.encoded_data[k] == static_cast<int>(i)) {
                    indices.push_back(k);
                }
            }
            results.emplace_back(dict_entry, indices);
        }
    }
    return results;
}

// Prefix query with no simd
vector<string> prefix_query_vanilla(const vector<string> data, const string& prefix) {
    vector<string> results;
    size_t prefix_length = prefix.size();
    if (prefix_length == 0) {
        cerr << "Error: Prefix length cannot be zero.\n";
        return results;
    }
    for (const auto& str : data) {
        if (str.size() >= prefix_length && str.compare(0, prefix_length, prefix) == 0) {
            results.push_back(str);
        }
    }
    return results;
}

// SIMD for singular item
vector<int> query_simd(const Encoded_Column& encoded_column, const string& query) {
    vector<int> indices;
    auto it = encoded_column.dictionary.data_to_id.find(query);
    if (it == encoded_column.dictionary.data_to_id.end()) {
        return indices; 
    }

    int query_id = it->second;
    size_t data_size = encoded_column.encoded_data.size();
    size_t simd_width = 8; 
    __m256i query_vec = _mm256_set1_epi32(query_id);
    size_t i = 0;
    for (; i + simd_width <= data_size; i += simd_width) {
        __m256i data_vec = _mm256_loadu_si256((__m256i*)&encoded_column.encoded_data[i]);
        __m256i cmp = _mm256_cmpeq_epi32(data_vec, query_vec);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
        for (int j = 0; j < simd_width; ++j) {
            if (mask & (1 << j)) {
                indices.push_back(i + j);
            }
        }
    }
    for (; i < data_size; ++i) {
        if (encoded_column.encoded_data[i] == query_id) {
            indices.push_back(i);
        }
    }
    return indices;
}

// vanilla search for singular item
vector<int> vanilla_search(const vector<string>& raw_data, const string& query) {
    vector<int> result_indices;
    for (size_t i = 0; i < raw_data.size(); ++i) {
        if (raw_data[i] == query) {
            result_indices.push_back(i);
        }
    }
    return result_indices;
}

int main() {
    string input_file = "Column.txt";
    string output_file = "encoded_data.txt";
    int num_threads ; 
    string query;
    cout << "Number of threads?" << endl;
    cin >> num_threads;
    auto data = read_column_from_file(input_file);
    auto start = chrono::high_resolution_clock::now();
    auto encoded_column = encode_dict(data, num_threads);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Encoding time: " << elapsed.count() << " s\n";

    encoded_column_to_file(encoded_column, output_file);
    string selection;
    bool quitting = false;
    while (quitting  == false){
        cout << "Singular search (s) or Prefix search (p)? (Type q to quit the program)" << endl;
        cin >> selection;
        if (selection == "q"){
            quitting = true;
        }
        else if (selection == "s"){ 
            cout << "Please enter a query for what you want to search for" << endl;
            cin >> query;
            cout << "\nTesting SIMD Single Query for: " << query << endl;
            auto start1 = chrono::high_resolution_clock::now();
            auto simd_single_results = query_simd(encoded_column, query);
            auto end1 = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed1 = end1 - start1;

            cout << "SIMD Single Query Results: ";
            for (int idx : simd_single_results) {
                cout << idx << " ";
            }
            cout << endl;

            auto start2 = chrono::high_resolution_clock::now();
            auto vanilla_single_results = vanilla_search(data, query);
            auto end2 = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed2 = end2 - start2;
            cout << "SIMD single search query time: " << elapsed1.count() << " s\n";
            cout << "Vanilla single search query time: " << elapsed2.count() << " s\n";

        }
        else if (selection == "p"){
            string prefix ; 
            cout << "Type your prefix" << endl;
            cin >> prefix;
            cout << "\nTesting SIMD Prefix Query for prefix: " << prefix << endl;
            auto start3 = chrono::high_resolution_clock::now();
            auto simd_prefix_results = prefix_query_simd(encoded_column, prefix);
            auto end3 = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed3 = end3 - start3;
            
            cout << "SIMD Prefix Query Results: ";
            for (const auto& result : simd_prefix_results) {
                const string& dict_entry = result.first;  
                const vector<int>& indices = result.second;  
                cout << "Data: " << dict_entry << " with indices: ";
                for (int idx : indices) {
                    cout << idx << " ";
                }
                cout << "\n";
            }
            cout << endl;

            auto start4 = chrono::high_resolution_clock::now();
            auto vanilla_prefix_results = prefix_query_vanilla(data, prefix);
            auto end4 = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed4 = end4 - start4;
            cout << "SIMD Prefix query time: " << elapsed3.count() << " s\n";
            cout << "Vanilla prefix query time: " << elapsed4.count() << " s\n";
        }
    }
}