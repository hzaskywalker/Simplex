#include "object.h"

namespace simplex{
    Shape::Shape(const CollisionObject& object): object(object){}

    CollisionObject* Shape::get_collision_object(size_t index){
        Matrix3d m;
        Vector3d vec;
        size_t start_index = index * 16;
        for(size_t i=0;i<3;++i){
            vec(i) = (*transforms)[start_index + i * 4 + 3];
            for(size_t j=0;j<3;++j){
                m(i, j) = (*transforms)[start_index + i * 4 + j];
            }
        }
        object.setRotation(m);
        object.setTranslation(vec);
        return &object;
    }

    int Shape::get_batch_size(){
        if(transforms.get()==0)
            return 0;
        return transforms->size()/16;
    }

    const Transforms3d& Shape::get_pose(){
        return *transforms;
    }

    void Shape::set_pose(const Transforms3d& _transforms){
        transforms = TransformsPtr(new Transforms3d(_transforms));
    }
}