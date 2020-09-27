import random

INPUT_LIST = [1, 2, 3, 4, 5]
TARGET_LIST = [1, 4, 9, 16, 25]
CONVERGENCE_FACTOR = .5


def matrix_multiply(list_1, weight_mat):
    """This function multiplies a 1D list by a 5x5 matrix to return a 5x1 list"""
    assert len(list_1) == len(weight_mat)
    assert len(weight_mat[0]) == 5

    output_list = []
    for i in range(5):
        temp = [sum([item * list_1[i]]) for item in weight_mat[i]]
        output_val = transformation_function(temp[0])
        output_list.append(output_val)

    assert len(output_list) == 5
    return output_list


def transformation_function(num):
    """This function transforms output values to create a non-linear model"""
    return num


def calc_sse(output_list, target_list):
    """This function calculates and returns the sum of the squared error between the calculated outputs
     and the targets"""
    sq_error = 0
    assert len(output_list) == len(target_list)

    for i in range(len(output_list)):
        sq_error += (output_list[i] - target_list[i]) ** 2

    assert sq_error > 0
    return sq_error


def create_weights():
    """This function creates and returns a 5x5 matrix with random floats between 0 and 1"""
    weight_mat = []
    for i in range(5):
        mat_row = []
        for j in range(5):
            mat_row.append(random.random())
        weight_mat.append(mat_row)

    assert len(weight_mat) == 5
    assert len(weight_mat[0]) == 5

    return weight_mat


def update_weights(weight_list_1, weight_list_2, output_list_1, output_list_2):
    """This function updates and returns both weight matrices by subtracting the weight functions' partial derivative
    with respect to the sum squared error function multiplied by the convergence factor"""

    assert len(weight_list_1) == 5
    assert len(weight_list_2) == 5
    assert len(weight_list_1[0]) == 5
    assert len(weight_list_2[0]) == 5

    return weight_list_1, weight_list_2


def main():
    weight_mat_1 = create_weights()
    weight_mat_2 = create_weights()
    sse = 1000
    while sse > .001:
        output_list_1 = matrix_multiply(INPUT_LIST, weight_mat_1)
        output_list_2 = matrix_multiply(output_list_1, weight_mat_2)
        sse = calc_sse(output_list_2, TARGET_LIST)
        weight_mat_1, weight_mat_2 = update_weights(weight_mat_1, weight_mat_2, output_list_1, output_list_2)


if __name__ == '__main__':
    main()
