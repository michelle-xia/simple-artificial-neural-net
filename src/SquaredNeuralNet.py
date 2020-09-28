import random
from math import exp

INPUT_LIST = [1, 2, 3, 4, 5]
INPUT_MAT = [[1], [2], [3], [4], [5]]
TARGET_LIST = [1, 4, 9, 16, 25]
CONVERGENCE_FACTOR = 20


def matrix_multiply(m1, m2):
    return [[sum(row * col for row, col in zip(m1_r, m2_c)) for m2_c in zip(*m2)] for m1_r in m1]


def transpose_matrix(mat):
    """This function transposes a 1 dimensional matrix and returns the result"""
    assert len(mat) > 0

    transposed_mat = []
    if type(mat[0]) == list:
        for element in mat:
            transposed_mat.append(element[0])
    else:
        for num in mat:
            temp = [num]
            transposed_mat.append(temp)
    return transposed_mat


def transformation_function(num):
    """This function transforms output values to create a non-linear model"""
    num = 1 / (1 + exp(-num))
    return num


def transformation_function_derivative(num):
    """This function computes the derivative for backpropagation"""
    num *= (1-num)
    return num


def calc_sse(output_list):
    """This function calculates and returns the sum of the squared error between the calculated outputs
     and the targets"""
    sq_error = 0
    assert len(output_list) == len(TARGET_LIST)

    for i in range(len(output_list)):
        sq_error += (output_list[i] - TARGET_LIST[i]) ** 2
    sq_error = 0.5 * sq_error
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


def update_weights(weight_mat_1, weight_mat_2, a_list_2, output_list):
    """This function updates and returns both weight matrices by subtracting the sse functions' partial derivative
    with respect to the weights multiplied by the convergence factor"""
    assert len(weight_mat_1) == 5
    assert len(weight_mat_2) == 5
    assert len(weight_mat_1[0]) == 5
    assert len(weight_mat_2[0]) == 5
    assert len(a_list_2) == 5  # a_list_2 is 1 x 5

    # convert to 2d list
    a_list_2 = [a_list_2]

    # compute Jacobian gradient for second weight matrix
    a3_deriv_list = [transformation_function_derivative(val) for val in output_list]  # derivative a3 with respect to z3

    # E with respect to z3
    delta_list_3 = [(-1 / len(INPUT_LIST)) * (TARGET_LIST[i] - output_list[i]) * a3_deriv_list[i]
                    for i in range(len(TARGET_LIST))]

    # transpose delta_list_3 to 5 x 1
    delta_mat_3 = transpose_matrix(delta_list_3)

    # compute final Jacobian gradient values for second matrix
    jacobian_list_2 = matrix_multiply(delta_mat_3, a_list_2)

    # convert to 2d list
    delta_list_3 = [delta_list_3]

    # compute Jacobian gradient for the first weight matrix
    delta_list_2 = matrix_multiply(delta_list_3, weight_mat_2)  # derivative of E with respect to a2

    # derivative a2 with respect to z2
    a2_deriv_list = [transformation_function_derivative(val) for val in a_list_2[0]]

    # E with respect to z2
    delta_list_2 = [a2 * z2 for a2 in delta_list_2[0] for z2 in a2_deriv_list]

    # transpose delta_list_2
    delta_mat_2 = transpose_matrix(delta_list_2)

    # compute final Jacobian gradient values for the first matrix
    jacobian_list_1 = matrix_multiply(delta_mat_2, INPUT_MAT)

    # reshape jacobian list
    jacobian_list_1 = reshape(jacobian_list_1)

    # update all weights
    weight_mat_2 = [[weight_mat_2[i][j] - jacobian_list_2[i][j] * CONVERGENCE_FACTOR for j in range(
        len(weight_mat_2[i]))]for i in range(len(weight_mat_2))]

    weight_mat_1 = [[weight_mat_1[i][j] - jacobian_list_1[i][j] * CONVERGENCE_FACTOR for j in range(
        len(weight_mat_1[i]))] for i in range(len(weight_mat_1))]

    return weight_mat_1, weight_mat_2


def print_output(output_list, sse):
    print("output list is", output_list)
    print("sse is", sse)
    print()


def reshape(mat):
    mat_len = int(len(mat) ** .5)
    reshaped_mat = []

    for i in range(mat_len):
        row = []
        for j in range(mat_len):
            row.append(mat[i * j][0])
        reshaped_mat.append(row)
    return reshaped_mat


def write_output(mat, num):
    file_name = 'weight_matrix_' + num + ".txt"
    with open(file_name, 'w') as writer:
        for row in mat:
            writer.write(''.join(row) + "\n")


def main():
    weight_mat_1 = create_weights()  # 5 x 5
    weight_mat_2 = create_weights()  # 5 x 5
    output_list = []
    sse = 1000
    epoch = 0
    while sse > .001:
        # compute values at layer 2
        z_mat_2 = matrix_multiply(INPUT_MAT, weight_mat_1)  # creates 5 x 5
        z_list_2 = transpose_matrix(z_mat_2)  # creates 1 x 5
        a_list_2 = [transformation_function(z2) for z2 in z_list_2]  # 1 x 5
        a_mat_2 = transpose_matrix(a_list_2)  # 5 x 1

        # compute output values
        z_mat_3 = matrix_multiply(weight_mat_2, a_mat_2)  # 5 x 1
        z_list_3 = transpose_matrix(z_mat_3)  # 1 x 5
        output_list = [transformation_function(z3) for z3 in z_list_3]  # 1 x 5

        # compute SSE
        sse = calc_sse(z_list_2)
        weight_mat_1, weight_mat_2 = update_weights(weight_mat_1, weight_mat_2, a_list_2, output_list)

        epoch += 1
        print("completed epoch", epoch)
        print_output(output_list, sse)

    print("\nFinal result:")
    print_output(output_list, sse)
    write_output(weight_mat_1, "1")
    write_output(weight_mat_2, "2")


if __name__ == '__main__':
    main()