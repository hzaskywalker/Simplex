#include "simplex.h"

namespace simplex{
    Simplex::Simplex(double contact_threshold): contact_threshold(contact_threshold){
    }
    Simplex::~Simplex(){
        for(auto shape_ptr=shapes.begin();shape_ptr!=shapes.end();++shape_ptr){
            // this might be very dangerous
            delete *shape_ptr;
        }
    }

    Shape* Simplex::add_box(double x, double y, double z){
        return add_shape(CollisionGeometryPtr_t(new fcl::Box<double>(x, y, z)));
    }

    Shape* Simplex::add_sphere(double R){
        return add_shape(CollisionGeometryPtr_t(new fcl::Sphere<double>(R)));
    }

    //Shape* Simplex::add_capsule(double R){
    //    return add_shape(CollisionGeometryPtr_t(new fcl::Sphere<double>(R)));
    //}

    Shape* Simplex::add_shape(CollisionGeometryPtr_t geom_ptr){
        auto shape_ptr = new Shape(geom_ptr);
        shapes.push_back(shape_ptr);
        return shape_ptr;
    }

    int Simplex::get_batch_size(){
        if(shapes.size()==0) return 0;
        int n = shapes.size();
        int batch_size = shapes[0]->get_batch_size();
        for(int i=0;i<n;++i){
            if(shapes[i]->get_batch_size()!=batch_size)
                throw std::runtime_error("Batch size of all objects should be the same");
        }
        return batch_size;
    }

    void Simplex::collide(){
        np.clear();
        batch.clear();
        dist.clear();
        collide_idx.clear();

        int n = shapes.size();

        fcl::CollisionRequest<double> collisionRequest(1, true);
        collisionRequest.num_max_contacts = 4;
        collisionRequest.gjk_solver_type = fcl::GST_LIBCCD;
        fcl::CollisionResult<double> collisionResult;

        if(n!=0){
            auto batch_size = get_batch_size();
            for(int batch_id=0; batch_id<batch_size; ++batch_id){
                vector<CollisionObject*> objects;
                for(int i=0;i<n;++i){
                    objects.push_back(shapes[i]->get_collision_object(0));
                }
                for(int i=0;i<n;++i){
                    for(int j=i+1;j<n;++j){
                        fcl::collide(objects[i], objects[j], collisionRequest, collisionResult);
                        if(collisionResult.isCollision()){
                            std::vector<fcl::Contact<double>> contacts;
                            collisionResult.getContacts(contacts);

                            for(auto contact=contacts.begin();contact!=contacts.end();++contact){
                                batch.push_back(batch_id);
                                for(size_t k=0;k<3;++k){
                                    np.push_back(contact->normal[k]);
                                }
                                for(size_t k=0;k<3;++k){
                                    np.push_back(contact->pos[k]);
                                }
                                dist.push_back(contact->penetration_depth);
                                collide_idx.push_back(i);
                                collide_idx.push_back(j);
                            }
                        }
                    }
                }
            }
        }
    }
}