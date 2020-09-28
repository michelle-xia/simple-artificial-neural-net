import random
from math import exp

INPUT_LIST = [1, 2, 3, 4, 5]
INPUT_MAT = [[1], [2], [3], [4], [5]]
TARGET_LIST = [1, 4, 9, 16, 25]
# weight_mat_1 = [[0 for j in range(0, 5)] for i in range(0, 5)]
# weight_mat_2 = [[0 for j in range(0, 5)] for i in range(0, 5)]

CONVERGENCE_FACTOR = .5


def matrix_multiply(m1, m2):
    """This function computes general maxtrix multiplication and returns the resulting matrix"""
    assert len(m1) > 0
    assert len(m2) > 0
    # assert len(m1[0]) == len(m2)

    if type(m1[0]) == list:
        result = [[0 for i in range(len(m2[0]))] for j in range(len(m1))]
        # row by row in m1
        for i in range(len(m1)):
            # column by column in m2
            for j in range(len(m2[i])):
                # row by row in m2
                for k in range(len(m2)):
                    result[i][j] += m1[i][k] * m2[k][j]

    else:
        result = [0 for i in range(len(m2[0]))]
        for i in range(len(m2[0])):
            for j in range(len(m1)):
                result[i] += m1[j] * m2[j][i]

    return result


def matrix_multiply_column(m1, m2):
    result = [[0 for i in range(len(m2))] for j in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2)):
            result[i][j] += m1[j][0] * m2[j]
    return result


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
    """This function updates and returns both weight matrices by subtracting the weight functions' partial derivative
    with respect to the sum squared error function multiplied by the convergence factor"""
    assert len(weight_mat_1) == 5
    assert len(weight_mat_2) == 5
    assert len(weight_mat_1[0]) == 5
    assert len(weight_mat_2[0]) == 5
    assert len(a_list_2) == 5  # a_list_2 is 1 x 5

    # compute Jacobian gradient for second weight matrix
    a3_deriv_list = [transformation_function_derivative(val) for val in output_list]  # derivative a3 with respect to z3

    # E with respect to z3
    delta_list_3 = [(-1 / len(INPUT_LIST)) * (y1 - val) * a for y1 in TARGET_LIST for val in output_list for a
                    in a3_deriv_list]

    # transpose delta_list_3 to 5 x 1
    delta_mat_3 = transpose_matrix(delta_list_3)

    # compute final Jacobian gradient values for second matrix
    jacobian_list_2 = matrix_multiply_column(delta_mat_3, a_list_2)

    print("deltal3", delta_list_3)

    # compute Jacobian gradient for the first weight matrix
    delta_list_2 = matrix_multiply(delta_list_3, weight_mat_2)  # derivative of E with respect to a2

    # derivative a2 with respect to z2
    a2_deriv_list = [transformation_function_derivative(val) for val in a_list_2]

    # E with respect to z2
    delta_list_2 = [a2 * z2 for a2 in delta_list_2 for z2 in a2_deriv_list]

    # transpose delta_list_2
    delta_mat_2 = transpose_matrix(delta_list_2)

    # compute final Jacobian gradient values for the first matrix
    jacobian_list_1 = matrix_multiply_column(delta_mat_2, INPUT_MAT)

    # update all weights
    weight_mat_2 = [[weight_mat_2[i][j] - jacobian_list_2[i][j] * CONVERGENCE_FACTOR for j in range(
        len(weight_mat_2[i]))]for i in range(len(weight_mat_2))]
    weight_mat_1 = [[weight_mat_1[i][j] - jacobian_list_1[i][j] * CONVERGENCE_FACTOR for j in range(
        len(weight_mat_1[i]))] for i in range(len(weight_mat_1))]

    return weight_mat_1, weight_mat_2


def main():
    weight_mat_1 = create_weights()
    weight_mat_2 = create_weights()
    output_list = []
    sse = 1000
    epoch = 0
    while sse > .001:
        z_mat_2 = matrix_multiply_column(INPUT_MAT, weight_mat_1)
        z_list_2 = transpose_matrix(z_mat_2)
        a_list_2 = [transformation_function(z2) for z2 in z_list_2]
        a_mat_2 = transpose_matrix(a_list_2)
        z_mat_3 = matrix_multiply(weight_mat_2, a_mat_2)
        z_list_3 = transpose_matrix(z_mat_3)
        output_list = [transformation_function(z3) for z3 in z_list_3]
        sse = calc_sse(z_list_2)
        weight_mat_1, weight_mat_2 = update_weights(weight_mat_1, weight_mat_2, a_list_2, output_list)
        epoch += 1
        print("completed epoch", epoch)
        print("output list is", output_list)
        print("\nweight matrix 1")
        for row in weight_mat_1:
            print(row)
        print("\nweight matrix 2")
        for row in weight_mat_2:
            print(row)

    print("\nFinal result:")
    print("output", output_list)
    print("weight matrix 1")
    for row in weight_mat_1:
        print(row)
    print("\nweight matrix 2")
    for row in weight_mat_2:
        print(row)


if __name__ == '__main__':
    main()
