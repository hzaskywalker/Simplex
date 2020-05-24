#pragma once
#include <Eigen/Dense>
#include "fcl/math/constants.h"
#include "fcl/narrowphase/collision.h"
#include "fcl/narrowphase/collision_object.h"
#include "object.h"

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
            ShapePtr capsule(double R, double l_x);
            //Shape* add_capsule(double R);
            //Shape* add_ground(double z=0);

            void collide();

            int get_batch_size();

            // collision result
            vector<int> batch;
            vector<double> dist;

            //normal and translation ..
            vector<double> np;
            vector<int> collide_idx;
            Simplex* add_shape(ShapePtr shape);
            void clear_shapes();
            int size();

        private:
            ShapePtr make_shape(CollisionGeometryPtr_t geom_ptr);

            double contact_threshold;
            vector<ShapePtr> shapes;
    };
} 