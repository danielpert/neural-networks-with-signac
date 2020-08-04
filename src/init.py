import signac
import itertools


if __name__ == '__main__':
    project = signac.init_project('neural-network')

    num_perceptrons = list(range(50, 155, 5))
    num_layers = list(range(1, 10))
    alpha = [1, 1e-1, 1e-2, 1e-3, 1e-4]

    for combin in itertools.product(num_perceptrons, num_layers, alpha):
        sp = {'num_perceptrons' : combin[0],
              'num_layers' : combin[1],
              'alpha' : combin[2]}
        job = project.open_job(sp)
        job.init()