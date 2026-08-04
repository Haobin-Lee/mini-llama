// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/radix_tree.h"

#include <algorithm>
#include <utility>

namespace mini_llama {

RadixTree::RadixTree() = default;
RadixTree::~RadixTree() = default;
RadixTree::RadixTree(RadixTree&&) noexcept = default;
RadixTree& RadixTree::operator=(RadixTree&&) noexcept = default;

void RadixTree::clear() {
    // 递归释放所有子节点到内存池
    auto clear_node = [this](auto& self, Node* node) -> void {
        for (auto& [key, child] : node->children) {
            self(self, child);
            pool_.deallocate(child);
        }
        node->children.clear();
    };
    clear_node(clear_node, &root_);

    root_.terminal = false;
    root_.key.clear();
    terminal_count_ = 0;
}

bool RadixTree::empty() const { return terminal_count_ == 0; }

size_t RadixTree::size() const { return terminal_count_; }

size_t RadixTree::commonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                     const std::vector<int>& b) {
    size_t i = 0;
    while (a_offset + i < a.size() && i < b.size() && a[a_offset + i] == b[i]) {
        ++i;
    }
    return i;
}

void RadixTree::insert(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        if (!root_.terminal) {
            root_.terminal = true;
            ++terminal_count_;
        }
        return;
    }
    insertInto(&root_, tokens);
}

void RadixTree::insertInto(Node* parent, const std::vector<int>& suffix) {
    auto child_it = parent->children.find(suffix[0]);
    // 没有匹配的子节点，直接创建新的子节点
    if (child_it == parent->children.end()) {
        auto child = pool_.allocate();
        child->key = std::move(suffix);
        child->terminal = true;
        parent->children[child->key[0]] = child;
        ++terminal_count_;
        return;
    }

    // 有匹配的子节点，继续递归插入
    Node* child = child_it->second;
    size_t common = commonPrefixLength(suffix, 0, child->key);
    if (common == child->key.size()) {
        // 完全匹配，则更新子节点为终端节点
        if (common == suffix.size()) {
            if (!child->terminal) {
                child->terminal = true;
                ++terminal_count_;
            }
            return;
        }
        // 部分匹配common <
        // suffix.size()，节点是输入的子集，继续递归插入剩余部分
        std::vector<int> remaining(suffix.begin() + static_cast<long>(common),
                                   suffix.end());
        insertInto(child, remaining);
        return;
    }

    // common < child->key.size()（不可能大于），则进行节点拆分
    auto split = pool_.allocate();
    split->key.assign(child->key.begin(),
                      child->key.begin() + static_cast<long>(common));

    child->key.erase(child->key.begin(),
                     child->key.begin() + static_cast<long>(common));
    int old_child_first = child->key[0];
    // 将不同的部分作为新的子节点
    split->children[old_child_first] = std::move(child);

    // 输入序列的suffix和child的前缀相同，则更新该节点为终端节点
    if (common == suffix.size()) {
        split->terminal = true;
        ++terminal_count_;
    } else {
        // 否则，将输入序列不同的部分作为新的子节点
        auto new_child = pool_.allocate();
        new_child->key.assign(suffix.begin() + static_cast<long>(common),
                              suffix.end());
        new_child->terminal = true;
        split->children[new_child->key[0]] = new_child;
        ++terminal_count_;
    }

    child_it->second = split;
}

size_t RadixTree::longestPrefix(const std::vector<int>& tokens) const {
    if (tokens.empty()) {
        return 0;
    }

    const Node* node = &root_;
    size_t pos = 0;
    while (pos < tokens.size()) {
        auto child_it = node->children.find(tokens[pos]);
        if (child_it == node->children.end()) {
            return pos;
        }

        const Node* child = child_it->second;
        size_t common = commonPrefixLength(tokens, pos, child->key);
        pos += common;
        if (common < child->key.size()) {
            return pos;
        }
        node = child;
    }
    return pos;
}

}  // namespace mini_llama
