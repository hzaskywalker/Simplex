#include "object.h"

namespace simplex{
    Shape::Shape(const CollisionObject& object_):object(object_){
        object.setIdentityTransform();
    }

    Shape::~Shape(){
        if(rot!=0){
            delete rot;
        } 
    }

    CollisionObject* Shape::get_collision_object(size_t index){
        if(transforms.get()!=0){
            // if we have set the pose
            Matrix3d m;
            Vector3d vec;
            size_t start_index = index * 16;
            if(transforms->size()==16)
                start_index = 0;
            for(size_t i=0;i<3;++i){
                vec(i) = (*transforms)[start_index + i * 4 + 3];
                for(size_t j=0;j<3;++j){
                    m(i, j) = (*transforms)[start_index + i * 4 + j];
                }
            }
            if(rot!=0){
                m = m * (*rot);
            }
            object.setRotation(m);
            object.setTranslation(vec);
        }
        return &object;
    }

    int Shape::get_batch_size(){
        if(transforms.get()==0)
            return 1;
        return transforms->size()/16;
    }

    const Transforms3d& Shape::get_pose(){
        return *transforms;
    }

    void Shape::set_pose(const Transforms3d& _transforms){
        transforms = TransformsPtr(new Transforms3d(_transforms));
    }
}