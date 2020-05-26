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

    ShapePtr Simplex::plane(double a, double b, double c, double d){
        return make_shape(CollisionGeometryPtr_t(new fcl::Plane<double>(a, b, c, d)));
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
        if(shapes.size()==0)
            return 0;
        int n = shapes.size();
        int batch_size = 1;
        for(int i=0;i<n;++i){
            int o_batch_size = shapes[i]->get_batch_size();
            if(o_batch_size!=1){
                if(batch_size!=1){
                    if(batch_size != o_batch_size)
                        throw std::runtime_error("Batch size of all objects should be the same");
                } else
                    batch_size = o_batch_size;
            }
        }
        return batch_size;
    }

    fcl::CollisionRequest<double> get_collision_request(){
        fcl::CollisionRequest<double> collisionRequest(1, true);
        collisionRequest.num_max_contacts = 4;
        collisionRequest.gjk_solver_type = fcl::GST_LIBCCD;
        return collisionRequest;
    }

    inline Eigen::MatrixXd Simplex::add_jacobian_column(const CollisionObject* a, const CollisionObject* b, double h, std::vector<fcl::Contact<double>>& contacts){

        // double length = 0.01
        collisionResult.clear();
        std::vector<fcl::Contact<double>> new_contacts;
        fcl::collide(a, b, collisionRequest, collisionResult);
        collisionResult.getContacts(new_contacts);

        auto s = a->getTranslation();
        Eigen::MatrixXd jac(contacts.size(), DIM);

        h = h * 10;

        for(size_t i=0;i<contacts.size();++i){
            auto& old_pos = contacts[i].pos;
            double nearest = 1e9;
            int new_id = 0;
            for(size_t j=0;j<new_contacts.size();++j){
                auto dist = (new_contacts[j].pos - old_pos).norm();
                if(dist < nearest){
                    nearest = dist;
                    new_id = j;
                }
            }
            if(nearest < h * (s-old_pos).norm()){
                // get the contacts ...
                jac(i, 0) = new_contacts[new_id].penetration_depth - contacts[i].penetration_depth;
                jac.row(i).segment(1,3) = new_contacts[new_id].normal - contacts[i].normal;
                jac.row(i).tail(3) = new_contacts[new_id].pos - contacts[i].pos;
            }
        }
        return jac;
    }

    void Simplex::compute_jacobian(CollisionObject* a, CollisionObject* b, double h, std::vector<fcl::Contact<double>>& contacts){
        // sign (2) x (rxa, vxa, rxb, vxb, rya, vya, ryb, vyb, rza, vza, rzb, vzb) x 2 x (1+3+3)
        // time complexity: 24 collision detection ...  
        // we should be able to optimize it 12 as we only care about the contact point??
        auto cc = collisionRequest.enable_cached_gjk_guess; //enable cache...
        collisionRequest.enable_cached_gjk_guess = true;
        collisionRequest.cached_gjk_guess = contacts[0].pos;

        auto rota = a->getRotation(), rotb = b->getRotation();
        auto veca = a->getTranslation(), vecb = b->getTranslation();

        Eigen::MatrixXd jac(contacts.size(), DIM * 24);

        auto n_c = contacts.size();

        int col = 0;
        for(int sign=0;sign<2;++sign){
            double si = sin(h * sign), ci = cos(h * sign);
            for (int i=0, j=1, k=2; i < 3; ++i, j=(j+1)%3, k=(k+1)%3){
                Matrix3d rot;
                rot(i,i)=ci; rot(i,j)=-si; rot(j,i)=si; rot(j,j)=ci; rot(k,k)=1;
                Vector3d vec; vec(i)=sign;

                a->setRotation(rota*rot);
                jac.block(0, col, n_c, DIM) = add_jacobian_column(a, b, h, contacts);
                a->setRotation(rota);

                a->setTranslation(veca+vec);
                jac.block(0, col+DIM, n_c, DIM) = add_jacobian_column(a, b, h, contacts);
                a->setTranslation(veca);

                b->setRotation(rotb*rot);
                jac.block(0, col+DIM*2, n_c, DIM) = add_jacobian_column(a, b, h, contacts);
                b->setRotation(rotb);

                b->setTranslation(vecb+vec);
                jac.block(0, col+DIM*3, n_c, DIM) = add_jacobian_column(a, b, h, contacts);
                b->setTranslation(vecb);

                col += 4*DIM;
            }
        }
        for(size_t i=0;i<n_c;++i){
            jacobian.push_back(jac.row(i));
        }

        collisionRequest.enable_cached_gjk_guess = cc;
    }

    void Simplex::backward(const Eigen::MatrixXd& dLdy){
        // dLdy is in the form of num_contact x 7
        // clear the grads ...
        for(size_t i=0;i<shapes.size();++i){
            shapes[i]->zero_grad();
        }
        int n_c = batch.size();
        int cc = 12 * DIM;
        Eigen::VectorXd ans;
        for(int i=0;i<n_c;++i){
            int batch_id = batch[i];
            for(int obj=0;obj<2;++obj){
                int obj_idx = 2 * i + obj;
                for(int d=0;i<VDIM;++d){//VDIM=6
                    int dim = (d%3) * 4 + obj * 2 + d/3;// a very stange low to get it..

                    double var = 0;
                    for(int sign=0;sign<2;++sign){
                        int l = dim * DIM + cc * sign;
                        double var2 = dLdy.row(i).dot(jacobian[i].segment(l, 7));
                        if(sign == 0 || var < var2){
                            var2 = var;
                        }
                    }
                    ans(d) = var;
                }
                shapes[obj_idx]->backward(batch_id, ans);
            }
        }
    }

    void Simplex::collide(bool computeJacobian, double epsilon){
        np.clear();
        batch.clear();
        dist.clear();
        collide_idx.clear();
        contact_id.clear();
        if(computeJacobian){
            jacobian.clear();
        }

        int n = shapes.size();

        collisionRequest = get_collision_request();

        if(n==0)
            return;

        auto batch_size = get_batch_size();
        for(int batch_id=0; batch_id<batch_size; ++batch_id){
            vector<CollisionObject*> objects;
            for(int i=0;i<n;++i){
                objects.push_back(shapes[i]->get_collision_object(batch_id));
            }
            int num_contact = 0;
            for(int i=0;i<n;++i){
                for(int j=i+1;j<n;++j){
                    if(!(shapes[i]->contype & shapes[j]->contype))
                        continue;
                    collisionResult.clear();
                    fcl::collide(objects[i], objects[j], collisionRequest, collisionResult);

                    if(collisionResult.isCollision()){
                        std::vector<fcl::Contact<double>> contacts;
                        collisionResult.getContacts(contacts);

                        for(auto contact=contacts.begin();contact!=contacts.end();++contact){
                            //XXX: this step seems to be very slow ... 
                            for(size_t k=0;k<3;++k){
                                np.push_back(contact->normal[k]);
                            }
                            for(size_t k=0;k<3;++k){
                                np.push_back(contact->pos[k]);
                            }
                            dist.push_back(contact->penetration_depth);
                            batch.push_back(batch_id);
                            collide_idx.push_back(i);
                            collide_idx.push_back(j);
                            contact_id.push_back(num_contact);

                            num_contact += 1;
                        }
                        if(computeJacobian){
                            compute_jacobian(objects[i], objects[j], epsilon, contacts);
                        }
                    }
                }
            }
        }
    }
}