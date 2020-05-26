#pragma once
#include <Eigen/Dense>
#include "fcl/math/constants.h"
#include "fcl/narrowphase/collision.h"
#include "fcl/narrowphase/collision_object.h"
#include "object.h"
#define DIM 7

using namespace std;
using CollisionGeometryPtr_t = std::shared_ptr<fcl::CollisionGeometry<double>>;

namespace simplex{
    using ShapePtr = std::shared_ptr<Shape>;

    class Simplex{
        public:
            explicit Simplex(double contact_threshold);
            ~Simplex();
            ShapePtr box(double x, double y, double z);
            ShapePtr sphere(double R=1);
            ShapePtr plane(double a, double b, double c, double d);
            ShapePtr capsule(double R, double l_x);
            //Shape* add_capsule(double R);
            //Shape* add_ground(double z=0);

            void collide(bool computeJacobian=false, double epsilon=1e-3);

            int get_batch_size();

            // collision result
            vector<int> batch;
            vector<int> contact_id;
            vector<double> dist;

            //normal and translation ..
            vector<double> np;
            vector<int> collide_idx;
            Simplex* add_shape(ShapePtr shape);
            void clear_shapes();
            int size();
            vector<Eigen::VectorXd> jacobian;
            void backward(const Eigen::MatrixXd& dLdy);

        private:
            ShapePtr make_shape(CollisionGeometryPtr_t geom_ptr);

            double contact_threshold;
            vector<ShapePtr> shapes;

            fcl::CollisionResultd collisionResult;
            fcl::CollisionRequestd collisionRequest;

            inline Eigen::MatrixXd add_jacobian_column(const CollisionObject* a, const CollisionObject* b, double h, std::vector<fcl::Contact<double>>& contacts);

            void compute_jacobian(CollisionObject* a, CollisionObject* b, double h, std::vector<fcl::Contact<double>>& contacts);
    };
} 