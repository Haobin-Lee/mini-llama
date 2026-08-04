// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_NODE_MEMORY_POOL_H_
#define INCLUDE_MINI_LLAMA_NODE_MEMORY_POOL_H_

#include <memory>
#include <unordered_map>
#include <vector>

namespace mini_llama {

struct Node {
    std::vector<int> key;
    bool terminal = false;
    std::unordered_map<int, Node*> children;
};

// 内存池实现
class NodeMemoryPool {
public:
    explicit NodeMemoryPool(size_t block_size = 4096);
    ~NodeMemoryPool();

    // 禁用拷贝
    NodeMemoryPool(const NodeMemoryPool&) = delete;
    NodeMemoryPool& operator=(const NodeMemoryPool&) = delete;

    // 支持移动
    NodeMemoryPool(NodeMemoryPool&&) noexcept = default;
    NodeMemoryPool& operator=(NodeMemoryPool&&) noexcept = default;

    // 分配节点
    Node* allocate();

    // 释放节点（放回空闲列表）
    void deallocate(Node* node);

private:
    void expand();

    size_t block_size_;
    std::vector<Node*> blocks_;     // 已分配的内存块
    std::vector<Node*> free_list_;  // 空闲节点列表
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_NODE_MEMORY_POOL_H_
