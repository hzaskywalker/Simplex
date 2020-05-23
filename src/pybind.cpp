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

PYBIND11_MODULE(fcl, m) {
    //m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");

    auto PySimplex = py::class_<Simplex>(m, "Simplex");
    PySimplex.def(py::init<double>(), py::arg("contact_threshold")=0)
        .def(
            "add_box",
            [](Simplex &simplex, const py::array_t<double> &arr) {
                if (arr.ndim() != 1 || arr.shape(0) != 3){
                    throw std::runtime_error("box size can only be (3,)");
                }
                return simplex.add_box(*arr.data(0), *arr.data(1), *arr.data(2));
            },
            py::arg("size") = make_array<double>({0, 0, 0}), py::return_value_policy::reference)
        .def(
            "collide", &Simplex::collide
        )
        .def_property_readonly(
          "batch", [](Simplex &simplex) {
              return py::array_t<int>(simplex.batch.size(), simplex.batch.data());
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
          "object_pair", [](Simplex& simplex) {
              return py::array_t<int>({int(simplex.batch.size()), 2}, simplex.collide_idx.data());
        })
        .def_property_readonly(
          "batch_size", &Simplex::get_batch_size
        );

    auto PyShape = py::class_<Shape>(m, "Shape");
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
        .def_property_readonly("batch_size", &Shape::get_batch_size);
}