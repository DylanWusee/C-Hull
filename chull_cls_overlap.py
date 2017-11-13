import numpy as np
import cvxpy as cvx
import scipy.io
import matplotlib.pyplot as plt

filename_train = './overlap_case/train_overlap.mat'
filename_test = './overlap_case/test_overlap.mat'
train_data = scipy.io.loadmat(filename_train)
test_data = scipy.io.loadmat(filename_test)

d = 0.9

A = np.array(train_data['A'])
B = np.array(train_data['B'])

_, num_samples = np.shape(A)

print "num_samples:", num_samples

u = cvx.Variable(num_samples)
v = cvx.Variable(num_samples)

constraints = [cvx.sum_entries(u) == 1,
              cvx.sum_entries(v) == 1,
              u >= 0,
              u <= d,
              v >= 0,
              v <= d]

obj = cvx.Minimize(cvx.sum_squares(A * u - B * v))

prob = cvx.Problem(obj, constraints)

prob.solve()

print "status: ", prob.status
print "optimal value " , prob.value
#print "optimal var", u.value, v.value

w = A * u.value - B * v.value
r = np.transpose(A * u.value + B * v.value) / 2.0 * w

print w
print r

X_data = np.array(test_data['X_test'])
test_label = np.squeeze(np.array(test_data['true_labels']))

_, num_test = np.shape(X_data)

print "num_test: ", num_test

test_res = np.array(w.T * X_data - r)
test_res = np.squeeze(test_res.T, axis = 1)

correct_count = 0
test_color = []
for i in range(0, num_test):
    if test_res[i] * test_label[i] > 0:
        correct_count += 1
    if test_label[i] < 0:
        test_color.append('b')
    else:
        test_color.append('r')

error = 1.0 - 1.0 * correct_count / num_test

print "classification error : ", error

# visualize the result
line_start_y = 10.0
line_end_y = -10.0
line_start_x = (r[0, 0] - w[1, 0] * line_start_y) / w[0, 0]
line_end_x = (r[0, 0] - w[1, 0] * line_end_y) / w[0, 0]
plt.figure('train_data')
plt.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 'g', label = 'classifier')
plt.scatter(A[0, :], A[1, :], c = 'r', label = 'A')
plt.scatter(B[0, :], B[1, :], c = 'b', label = 'B')
plt.legend()
plt.axis([-10, 10, -10, 10])
plt.title('Train Data')

plt.figure('test_data')
plt.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 'g', label = 'classifier')
plt.scatter(X_data[0, :], X_data[1, :], c = test_color)
plt.legend()
plt.axis([-7, 7, -7, 7])
plt.title('Test Data classification error : ' + str(error))

plt.show()







