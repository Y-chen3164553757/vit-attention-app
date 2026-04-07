# Copilot 助理说明

> 🌐 Language / 语言：**中文** | [English](#english-version)

---

## 本仓库中 Copilot 助理的身份与职责

### 它是谁？

本仓库使用了 **GitHub Copilot**（以下简称"助理"）作为开发辅助工具。  
助理是一个由 AI 驱动的编程助手，可在对话中回答技术问题、解释代码逻辑、提供实现建议，以及根据明确请求帮助生成代码或文档。

### 它能做什么？

- 在对话中解答有关本仓库代码的技术问题（如架构、算法、依赖等）
- 根据用户描述生成代码片段或文档草稿，供开发者参考
- 解释 ViT 注意力机制、DINOv3 模型、FastAPI 接口等相关技术原理
- **在用户明确发出请求时**，创建 Pull Request 并将变更提交至仓库

### 它不会做什么？

- ❌ **不会替代人工代码审核（Code Review）**：助理提供的所有建议均需由开发者自行判断和验证，不构成最终决策。
- ❌ **不会自动向仓库写入内容**：除非用户明确要求助理"创建 PR"或"提交更改"，否则助理不会主动修改、提交或推送任何代码。
- ❌ **不具备仓库管理权限**：助理无法自主合并 PR、管理分支保护规则或变更仓库设置。
- ❌ **不保证输出的完全正确性**：AI 生成的内容可能存在错误或不适用于特定场景，请始终进行人工复核。

### 使用建议

1. **明确说明意图**：在与助理对话时，请清楚描述你的需求，例如"帮我解释这段代码"或"请创建一个 PR 来完成 X 功能"。
2. **人工复核优先**：将助理的输出视为草稿或参考，而非可直接使用的最终成果。
3. **谨慎授权**：仅在你确认变更内容后，再明确要求助理提交代码或创建 PR。

---

## English Version

### Who is the assistant?

This repository uses **GitHub Copilot** (referred to as "the assistant") as a development aid.  
The assistant is an AI-powered coding helper that can answer technical questions, explain code logic, provide implementation suggestions, and help generate code or documentation upon explicit request.

### What can it do?

- Answer technical questions about this repository's code (architecture, algorithms, dependencies, etc.) during conversations
- Generate code snippets or documentation drafts for developers to review
- Explain concepts such as ViT self-attention, DINOv3, FastAPI, and related topics
- **Create Pull Requests and commit changes to the repository when explicitly requested by the user**

### What it will NOT do?

- ❌ **Replace human code review**: All suggestions provided by the assistant must be evaluated and verified by the developer. They do not constitute final decisions.
- ❌ **Automatically write to the repository**: The assistant will not modify, commit, or push any code unless the user explicitly asks it to "create a PR" or "submit changes."
- ❌ **Act as a repository administrator**: The assistant cannot autonomously merge PRs, manage branch protection rules, or change repository settings.
- ❌ **Guarantee the correctness of its output**: AI-generated content may contain errors or may not be suitable for specific scenarios. Always perform a human review.

### Recommendations

1. **Be explicit about your intent**: Clearly describe what you need, e.g., "Explain this code" or "Create a PR to implement feature X."
2. **Human review first**: Treat the assistant's output as a draft or reference, not a ready-to-use final result.
3. **Authorize carefully**: Only ask the assistant to commit code or create a PR after you have reviewed and confirmed the changes.
