
// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/node_memory_pool.h"

namespace mini_llama {
// 内存池实现

NodeMemoryPool::NodeMemoryPool(size_t block_size) : block_size_(block_size) {}

NodeMemoryPool::~NodeMemoryPool() {
    for (auto& block : blocks_) {
        delete[] block;
    }
}

// 分配节点
Node* NodeMemoryPool::allocate() {
    if (free_list_.empty()) {
        expand();
    }
    Node* node = free_list_.back();
    free_list_.pop_back();
    return node;
}

// 释放节点（放回空闲列表）
void NodeMemoryPool::deallocate(Node* node) {
    // 重置节点状态
    node->key.clear();
    node->terminal = false;
    node->children.clear();
    free_list_.push_back(node);
}

void NodeMemoryPool::expand() {
    // 分配一块包含多个 Node 的连续内存
    size_t nodes_per_block = block_size_ / sizeof(Node);
    if (nodes_per_block == 0) {
        nodes_per_block = 1;
    }

    if (blocks_.empty() && free_list_.empty()) {
        blocks_.reserve(nodes_per_block);
        free_list_.reserve(nodes_per_block);
    }
    Node* block = new Node[nodes_per_block];
    blocks_.push_back(block);

    // 将新分配的节点加入空闲列表（逆序，保证顺序分配）
    for (size_t i = nodes_per_block; i > 0; --i) {
        free_list_.push_back(&block[i - 1]);
    }
}

}  // namespace mini_llama