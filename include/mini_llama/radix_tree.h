// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_RADIX_TREE_H_
#define INCLUDE_MINI_LLAMA_RADIX_TREE_H_

#include <map>
#include <memory>
#include <vector>

#include "mini_llama/node_memory_pool.h"

namespace mini_llama {

// Compressed prefix tree for token sequences.
class RadixTree {
public:
    RadixTree();
    ~RadixTree();

    RadixTree(const RadixTree&) = delete;
    RadixTree& operator=(const RadixTree&) = delete;
    RadixTree(RadixTree&&) noexcept;
    RadixTree& operator=(RadixTree&&) noexcept;

    void clear();
    bool empty() const;
    size_t size() const;

    void insert(const std::vector<int>& tokens);
    size_t longestPrefix(const std::vector<int>& tokens) const;

private:
    static size_t commonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                     const std::vector<int>& b);
    void insertInto(Node* parent, const std::vector<int>& suffix);

    Node root_;
    size_t terminal_count_ = 0;
    NodeMemoryPool pool_;  // 内存池
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_RADIX_TREE_H_
