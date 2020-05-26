#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "simplex.h"

using namespace simplex;
namespace py=pybind11;

int add(int i, int j){
    return i+j;
}

template <typename T> py::array_t<T> make_array(std::vector<T> const &values) {
  return py::array_t<T>(values.size(), values.data());
}

PYBIND11_MODULE(simplex_c, m) {
    //m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");

    auto PySimplex = py::class_<Simplex>(m, "Simplex");
    PySimplex.def(py::init<double>(), py::arg("contact_threshold")=0)
        .def(
            "box",
            [](Simplex &simplex, const py::array_t<double> &arr) {
                if (arr.ndim() != 1 || arr.shape(0) != 3){
                    throw std::runtime_error("box size can only be (3,)");
                }
                return simplex.box(*arr.data(0), *arr.data(1), *arr.data(2));
            },
            py::arg("size") = make_array<double>({0, 0, 0}), py::return_value_policy::reference)
        .def(
            "sphere", &Simplex::sphere,
            py::arg("R") = 1.0, py::return_value_policy::reference)
        .def(
            "capsule", &Simplex::capsule,
            py::arg("R") = 0.1, py::arg("l_x")=1., py::return_value_policy::reference)
        .def(
            "plane", &Simplex::plane,
            py::arg("a") = 0, py::arg("b")=0, py::arg("c")=0, py::arg("d")=0, py::return_value_policy::reference)
        .def(
            "collide", &Simplex::collide, py::arg("computeJacobian")=false, py::arg("epsilon")=1e-3
        )
        .def(
            "add_shape", &Simplex::add_shape, py::return_value_policy::reference
        )
        .def(
            "clear_shapes", &Simplex::clear_shapes
        )
        .def_property_readonly(
          "batch", [](Simplex &simplex) {
              return py::array_t<int>(simplex.batch.size(), simplex.batch.data());
        })
        .def_property_readonly(
          "n", [](Simplex &simplex) {
              return simplex.size();
        })
        .def_property_readonly(
          "dist", [](Simplex &simplex) {
              return py::array_t<double>(simplex.dist.size(), simplex.dist.data());
        })
        .def_property_readonly(
          "normal_pos", [](Simplex& simplex) {
              return py::array_t<double>({int(simplex.batch.size()), 2, 3}, simplex.np.data());
        })
        .def_property_readonly(
          "contact_id", [](Simplex& simplex) {
              return py::array_t<int>(simplex.batch.size(), simplex.contact_id.data());
        })
        .def_property_readonly(
          "object_pair", [](Simplex& simplex) {
              return py::array_t<int>({int(simplex.batch.size()), 2}, simplex.collide_idx.data());
        })
        .def_property_readonly(
          "jacobian", [](Simplex& simplex){
              //only for testing ...
              int dim = 24 * 7;
              int nc = simplex.batch.size();
              Eigen::MatrixXd ans(nc, dim);
              for(int i=0;i<nc;++i){
                  ans.row(i) = simplex.jacobian[i];
              }
              return ans;
          }
        )
        .def_property_readonly(
          "batch_size", &Simplex::get_batch_size
        )
        .def(
            "backward", &Simplex::backward
        );

    auto PyShape = py::class_<Shape, std::shared_ptr<Shape>>(m, "Shape");
    PyShape.def(
               "set_pose",
               [](Shape &shape, const py::array_t<double> &arr) {
                   if (arr.ndim() != 3 || arr.shape(1) != 4 || arr.shape(2) != 4)
                       throw std::runtime_error("pose should have shape [batch_size, 4, 4]");
                   shape.set_pose(vector<double>(arr.data(), arr.data() + arr.size()));
               },
               py::arg("pose"))
        .def("get_pose", [](Shape &shape) {
            return py::array_t<double>({shape.get_batch_size(), 4, 4}, shape.get_pose().data());
        })
        .def_property_readonly("batch_size", &Shape::get_batch_size)
        .def_property_readonly("grad", [](Shape &shape){return shape.grads;})
        .def_readwrite("contype", &Shape::contype)
        .def("zero_grad", &Shape::zero_grad);
}