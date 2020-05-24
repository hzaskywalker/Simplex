#include "simplex.h"

namespace simplex{
    Simplex::Simplex(double contact_threshold): contact_threshold(contact_threshold){
    }

    Simplex::~Simplex(){
        clear_shapes();
    }

    ShapePtr Simplex::box(double x, double y, double z){
        return make_shape(CollisionGeometryPtr_t(new fcl::Box<double>(x, y, z)));
    }

    ShapePtr Simplex::sphere(double R){
        return make_shape(CollisionGeometryPtr_t(new fcl::Sphere<double>(R)));
    }

    ShapePtr Simplex::capsule(double R, double l_x){
        auto shape = make_shape(CollisionGeometryPtr_t(new fcl::Capsule<double>(R, l_x)));
        shape->rot = new Matrix3d();
        (*shape->rot) << 0,0,1,0,1,0,-1,0,0;
        return shape;
    }

    ShapePtr Simplex::make_shape(CollisionGeometryPtr_t geom_ptr){
        auto shape_ptr = ShapePtr(new Shape(geom_ptr));
        return shape_ptr;
    }

    Simplex* Simplex::add_shape(ShapePtr shape){
        shapes.push_back(shape);
        return this;
    }

    void Simplex::clear_shapes(){
        shapes.clear();
    }

    int Simplex::size(){
        return shapes.size();
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

        if(n!=0){
            auto batch_size = get_batch_size();
            for(int batch_id=0; batch_id<batch_size; ++batch_id){
                vector<CollisionObject*> objects;
                for(int i=0;i<n;++i){
                    objects.push_back(shapes[i]->get_collision_object(batch_id));
                }
                for(int i=0;i<n;++i){
                    for(int j=i+1;j<n;++j){
                        if(!(shapes[i]->contype & shapes[j]->contype))
                            continue;
                        fcl::CollisionResult<double> collisionResult;
                        fcl::collide(objects[i], objects[j], collisionRequest, collisionResult);

                        cout<<"num contacts "<<collisionResult.isCollision()<<endl;

                        if(collisionResult.isCollision()){
                            std::vector<fcl::Contact<double>> contacts;
                            collisionResult.getContacts(contacts);

                            for(auto contact=contacts.begin();contact!=contacts.end();++contact){
                                /*
                                cout<<"normal"<<endl;
                                cout<<contact->normal[0]<<" ";
                                cout<<contact->normal[1]<<" ";
                                cout<<contact->normal[2]<<endl;
                                */
                                //XXX: this step seems to be very slow ... 
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