
1. 最快方法直接 invalidate 那个 key

2. 传到 github 上, force 方法：
	1. 还是首先 revoke 掉那个 key
	2. 然后回退到上一个  commit，但是保留修改在工作区: `git reset --soft HEAD~1`
	3. 修改 credential 代码
	4. 重新 commit `git commmit -m "Remove XXX"`
	5. 强制推送: `git push --force`

3. 是比较早的 commit:
	1. 找到 commit 位置 `git log --oneline`
		- 查看提交历史，定位**包含凭据的那个 commit**
		- 记住它在当前 HEAD 往回数是第几个（例如倒数第 5 个）
	2. 假设修改的是倒数第五个 commit: `git rebase -i HEAD~5`
		- 含义：从 `HEAD~5` 这个基点开始，把**最近 5 个 commit 取出来重新重放**
		- `-i`（interactive）表示可以在重放过程中**人工干预某些 commit**
		- 这一步会打开一个 rebase 脚本文件，列出这 5 个 commit
	3. 把有问题的 commit 前面的 `pick` 改成 `edit`：`edit 123abc commit with credential`
		- `pick`：表示该 commit 原样重放
		- `edit`：表示重放到该 commit 时**暂停**，允许修改这个 commit 的内容
		- 目的：让 Git 在包含凭据的 commit 处停下来，方便清理泄露信息
	4. 修改代码：`git add .`, `git commit --amend`
		- 此时 Git 已经应用了该 commit，并暂停在这里
		- 修改代码，删除硬编码的 credential（改用环境变量等方式）
		- `git add .`：将修正后的文件加入暂存区
		- `git commit --amend`：**就地修改当前 commit**，用修正后的版本替换原来的有问题 commit
		- 该操作不会新建 commit，而是重写当前这个 commit
	5. rebase 代码：`git rebase --continue`
		- 告诉 Git：当前需要编辑的 commit 已修复完成
		- Git 会继续按照 rebase 脚本，将后续的 commit 依次重新应用
		- 如果出现冲突，需要手动解决后再执行 `git rebase --continue`
	6. 强制推送：`git push --force`
		- rebase 会导致 commit hash 改变，远端历史与本地不一致
		- 使用 `--force` 用**重写后的安全历史**覆盖远端历史
		- 实际操作前应确保凭据已经在服务端被 revoke / rotate

4. 更标准用 `git filter-repo`：
	1. 删除 commit history 中某个文件：`git filter-repo --path path/to/file --invert-paths`
	2. 删除特定字符串：`git filter-repo --replace-text secrets.txt`，这里文件的每一行类似于 `MY_SECRET_KEY==>REMOVED`
	3. 然后强制推送: `git push --force --all`, `git push --force --tags`

5. 以上做完，别人下载的时候需要  `git fetch --all && git reset --hard origin/main`



【git rebase -i 的“脑内动画”】【用于清理较早 commit 中的凭据】

- 原始提交历史（从旧到新）：
	... -> C0 -> C1 -> C2 -> C3(leak) -> C4 -> C5(HEAD)
- 其中：
	- C3 是包含硬编码凭据的 commit
	- HEAD 指向当前最新的 commit（C5）
- 执行命令：
	git rebase -i HEAD~5
- 含义：
	- 以 C0 作为新的基点
	- 将 C1 ~ C5 这 5 个 commit 取出来，准备“重新播放”一遍
- rebase 脚本阶段（交互编辑）：
	pick C1
	pick C2
	edit C3   ← 在此暂停（包含凭据的 commit）
	pick C4
	pick C5
- rebase 执行过程（时间顺序）：
	1. Git 重新应用 C1，生成 C1'
	2. Git 重新应用 C2，生成 C2'
	3. Git 应用 C3 后暂停（进入 edit 状态）
		- 工作区此时等价于 “刚提交完 C3 的状态”
		- 开发者修改代码，删除泄露的凭据
		- 执行 `git add .`
		- 执行 `git commit --amend`
		- 原来的 C3 被替换为安全版本 C3'
	4. 执行 `git rebase --continue`
		- Git 继续重放后续提交
		- 依次生成 C4'、C5'
- rebase 完成后的提交历史：
	... -> C0 -> C1' -> C2' -> C3' -> C4' -> C5'(HEAD)
- 关键结论：
	- C3' 与原来的 C3 是“逻辑上同一个提交”，但内容已被修复
	- C1' ~ C5' 的 commit hash 全部发生变化
	- Git 历史中不再存在任何包含凭据的 commit
- 后续操作：
	- 由于 commit hash 改变，必须执行 `git push --force`
	- 即使历史已清理，泄露的凭据仍需立即 revoke / rotate



所以最好一开始就 `.env` + `.gitignore`.