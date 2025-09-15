[Problems](https://leetcode.cn/discuss/post/3578981/ti-dan-hua-dong-chuang-kou-ding-chang-bu-rzz7/)

1. 固定窗口大小

一个循环，right element in -> update target -> left element out

[643](https://leetcode.cn/problems/maximum-average-subarray-i/description/)
```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        """
        Args:
            nums: given array
            k: fixed window size
        """ 
        ans = -float('inf')
        cur_sum = 0  # init target

        for i in range(len(nums)):
            cur_sum += nums[i]  # element in
            
            if i - k + 1 >= 0:
                ans = max(ans, cur_sum / k)  # update
                cur_sum -= nums[i - k + 1]  # element out
        return ans
```
> 注意当 `k = 0`，window 长度为 0 时，左边界 `i - k + 1` 在算最后一步的时候会超过 `len(nums) - 1`，所以需要提前判断剔除 [1423](https://leetcode.cn/problems/maximum-points-you-can-obtain-from-cards/)
