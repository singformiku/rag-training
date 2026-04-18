---
name: git-commit-writer
description: Generates Conventional-Commits-style git commit messages by analyzing staged changes (git diff --cached). Use whenever the user asks to write, draft, improve, or suggest a commit message, or mentions "commit message", "conventional commits", "git commit", or has staged changes and wants to commit. Do NOT use for writing PR descriptions or explaining past commit history.
license: MIT
---

# Git Commit Message Writer

## Workflow
1. 讀取 `git diff --cached`；若無 staged 改動，提示先 `git add`。
2. 判斷 type：feat/fix/docs/style/refactor/perf/test/chore
3. 判斷 scope：`(auth)`, `(api)`, `(ui)`
4. 寫 subject（祈使語氣、≤50 字元、小寫開頭、無句點）
5. 寫 body（>1 檔變更時，說明 why 而非 what，72 字元換行）
6. 寫 footer（選填）：`BREAKING CHANGE:` 或 `Closes #123`

## Examples

### feat
Input: 新增 auth/jwt.py 含 sign_token/verify_token
Output:
`feat(auth): add JWT sign and verify helpers`

(body: Introduces HS256-based token signing to replace legacy cookie flow...)

## Edge cases
- Mixed concerns（同時 feat + fix）→ 建議拆成兩個 commit
- Diff >500 行 → 建議拆分
- 沒有 staged → "No staged changes. Run `git add` first."
