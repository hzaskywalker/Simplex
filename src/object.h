#pragma once
#include <Eigen/Dense>
#include "fcl/math/constants.h"
#include "fcl/narrowphase/collision.h"
#include "fcl/narrowphase/collision_object.h"

using fcl::Matrix3d;
using fcl::Vector3d;
using fcl::Transform3d;
using namespace std;

typedef vector<double> Transforms3d;
using CollisionObject = fcl::CollisionObject<double>;
typedef unique_ptr<Transforms3d> TransformsPtr;

namespace simplex{
    class Shape{
        public:
            explicit Shape(const CollisionObject& geom);
            CollisionObject* get_collision_object(size_t index);
            int get_batch_size();

            void set_pose(const Transforms3d& transforms);
            const Transforms3d& get_pose();

            TransformsPtr transforms;
        private:
            CollisionObject object;
    };
}