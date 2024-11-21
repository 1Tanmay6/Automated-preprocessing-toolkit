def find_smallest_element(arr):
    """
    Finds the smallest element in an array.

    Args:
        arr (list): The input array.

    Returns:
        int: The smallest element in the array.
    """
    # Initialize the smallest element to be the first element in the array
    smallest = arr[0]
    
    # Loop through all elements in the array
    for num in arr:
        # If the current element is smaller than the current smallest element, update the smallest element
        if num < smallest:
            smallest = num
    
    return smallest

# Driver code to test the function
arr = [5, 3, 8, 1, 2]
smallest_element = find_smallest_element(arr)
print("The smallest element in the array is:", smallest_element)
