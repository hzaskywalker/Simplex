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
        Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(contacts.size(), DIM);

        double tolerance = abs(h * 10);

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
            if(nearest < tolerance * (s-old_pos).norm()){
                // get the contacts ...
                jac(i, 0) = new_contacts[new_id].penetration_depth - contacts[i].penetration_depth;
                jac.row(i).segment(1,3) = new_contacts[new_id].normal - contacts[i].normal;
                jac.row(i).tail(3) = new_contacts[new_id].pos - contacts[i].pos;
            }
        }
        return jac/h;
    }

    #define START(i, j, k) (((i)*VDIM*2 + (k)*VDIM + (j))*DIM)

    void Simplex::compute_jacobian(CollisionObject* a, CollisionObject* b, double h, std::vector<fcl::Contact<double>>& contacts){
        // sign obj(2) x sign(2) x 6 x (1+3+3)
        auto cc = collisionRequest.enable_cached_gjk_guess; //enable cache...
        collisionRequest.enable_cached_gjk_guess = true;
        collisionRequest.cached_gjk_guess = contacts[0].pos;

        auto rota = a->getRotation(), rotb = b->getRotation();
        auto veca = a->getTranslation(), vecb = b->getTranslation();

        Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(contacts.size(), DIM * 24);

        #define __ROW__(i,j,k) jac.block(0, START(i,j,k), n_c, DIM)


        auto n_c = contacts.size();

        for(int sign=0;sign<2;++sign){
            double s = sign * 2 - 1;
            double dx = s * h;
            double ci = cos(dx), si = sin(dx);
            for (int i=0, j=1, k=2; i < 3; ++i, j=(j+1)%3, k=(k+1)%3){
                Matrix3d rot = Matrix3d::Zero();
                rot(j,j)=ci; rot(j,k)=-si; rot(k,j)=si; rot(k,k)=ci; rot(i,i)=1;
                Vector3d vec = Vector3d::Zero();
                vec(i)=dx;
                // the i-th ball, j-th

                a->setRotation(rota*rot);
                __ROW__(0, i, sign) = add_jacobian_column(a, b, dx, contacts);
                a->setRotation(rota);

                a->setTranslation(veca+rota*vec);
                __ROW__(0, i+3, sign) = add_jacobian_column(a, b, dx, contacts);
                a->setTranslation(veca);

                b->setRotation(rotb*rot);
                __ROW__(1, i, sign) = add_jacobian_column(a, b, dx, contacts);
                b->setRotation(rotb);

                b->setTranslation(vecb+rotb*vec);
                __ROW__(1, i+3, sign) = add_jacobian_column(a, b, dx, contacts);
                b->setTranslation(vecb);
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
        if(jacobian.size()!=batch.size()){
            throw std::runtime_error("Can't backward the collide function if forward is not called");
        }
        for(size_t i=0;i<shapes.size();++i){
            shapes[i]->zero_grad();
        }
        int n_c = batch.size();
        Eigen::VectorXd ans = Eigen::VectorXd::Zero(VDIM);
        for(int i=0;i<n_c;++i){
            int batch_id = batch[i];
            for(int obj=0;obj<2;++obj){
                auto tmp = jacobian[i].segment(START(obj, 0, 0), 2*VDIM*DIM);
                Eigen::VectorXd tmp2 = (tmp.head(VDIM*DIM) + tmp.tail(VDIM*DIM))/2;
                Eigen::MatrixXd Jac = Eigen::Map<Eigen::MatrixXd>(tmp2.data(), DIM, VDIM);
                int obj_idx = collide_idx[2 * i + obj];
                shapes[obj_idx]->backward(batch_id, dLdy.row(i)*Jac);
            }
        }
    }

    void Simplex::collide(bool computeJacobian, double epsilon){
        np.clear();
        batch.clear();
        dist.clear();
        collide_idx.clear();
        contact_id.clear();
        jacobian.clear();

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