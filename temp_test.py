
def no_of_subsequences(nums, k):
    def count_subsequences_with_product_less_than_k(nums, target, product, index=0, current_product=1):
        if index == len(nums):
            return 0
        count = count_subsequences_with_product_less_than_k(nums, target, product, index + 1, current_product)
        if current_product * nums[index] < target:
            count += 1 + count_subsequences_with_product_less_than_k(nums, target, product, index + 1, current_product * nums[index])
        return count

    return count_subsequences_with_product_less_than_k(nums, k, 1)

# Example usage
print(no_of_subsequences([1, 2, 3], 10))  # Output: 8


if __name__ == "__main__":
    assert no_of_subsequences([1,2,3,4], 10) == 11
    assert no_of_subsequences([4,8,7,2], 50) == 9
    assert no_of_subsequences([5,6,7,8], 15) == 4
