
// Code generated by stanc v2.31.0
#include <stan/model/model_header.hpp>
namespace normalx462_model_namespace {

using stan::model::model_base_crtp;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 17> locations_array__ = 
{" (found before start of program)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 15, column 3 to column 33)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 16, column 3 to column 20)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 24, column 8 to column 55)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 23, column 20 to line 25, column 5)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 23, column 4 to line 25, column 5)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 2, column 2 to column 17)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 3, column 2 to column 17)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 4, column 2 to column 17)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 6, column 8 to column 9)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 6, column 2 to column 36)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 7, column 8 to column 9)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 7, column 2 to column 36)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 8, column 8 to column 9)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 8, column 2 to column 20)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 15, column 9 to column 10)",
 " (in '/home/rocco/crowd_geoloc_https/src/crowdgeoloc/cmdstan/cmdstan/normal.2.stan', line 16, column 9 to column 10)"};




class normalx462_model final : public model_base_crtp<normalx462_model> {

 private:
  int w;
  int t;
  int a;
  std::vector<int> t_A;
  std::vector<int> w_A;
  std::vector<double> ann; 
  
 
 public:
  ~normalx462_model() { }
  
  inline std::string model_name() const final { return "normalx462_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.31.0", "stancflags = "};
  }
  
  
  normalx462_model(stan::io::var_context& context__,
                   unsigned int random_seed__ = 0,
                   std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "normalx462_model_namespace::normalx462_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 6;
      context__.validate_dims("data initialization","w","int",
           std::vector<size_t>{});
      w = std::numeric_limits<int>::min();
      
      
      current_statement__ = 6;
      w = context__.vals_i("w")[(1 - 1)];
      current_statement__ = 6;
      stan::math::check_greater_or_equal(function__, "w", w, 1);
      current_statement__ = 7;
      context__.validate_dims("data initialization","t","int",
           std::vector<size_t>{});
      t = std::numeric_limits<int>::min();
      
      
      current_statement__ = 7;
      t = context__.vals_i("t")[(1 - 1)];
      current_statement__ = 7;
      stan::math::check_greater_or_equal(function__, "t", t, 1);
      current_statement__ = 8;
      context__.validate_dims("data initialization","a","int",
           std::vector<size_t>{});
      a = std::numeric_limits<int>::min();
      
      
      current_statement__ = 8;
      a = context__.vals_i("a")[(1 - 1)];
      current_statement__ = 8;
      stan::math::check_greater_or_equal(function__, "a", a, 1);
      current_statement__ = 9;
      stan::math::validate_non_negative_index("t_A", "a", a);
      current_statement__ = 10;
      context__.validate_dims("data initialization","t_A","int",
           std::vector<size_t>{static_cast<size_t>(a)});
      t_A = std::vector<int>(a, std::numeric_limits<int>::min());
      
      
      current_statement__ = 10;
      t_A = context__.vals_i("t_A");
      current_statement__ = 10;
      stan::math::check_greater_or_equal(function__, "t_A", t_A, 1);
      current_statement__ = 10;
      stan::math::check_less_or_equal(function__, "t_A", t_A, t);
      current_statement__ = 11;
      stan::math::validate_non_negative_index("w_A", "a", a);
      current_statement__ = 12;
      context__.validate_dims("data initialization","w_A","int",
           std::vector<size_t>{static_cast<size_t>(a)});
      w_A = std::vector<int>(a, std::numeric_limits<int>::min());
      
      
      current_statement__ = 12;
      w_A = context__.vals_i("w_A");
      current_statement__ = 12;
      stan::math::check_greater_or_equal(function__, "w_A", w_A, 1);
      current_statement__ = 12;
      stan::math::check_less_or_equal(function__, "w_A", w_A, w);
      current_statement__ = 13;
      stan::math::validate_non_negative_index("ann", "a", a);
      current_statement__ = 14;
      context__.validate_dims("data initialization","ann","double",
           std::vector<size_t>{static_cast<size_t>(a)});
      ann = std::vector<double>(a, std::numeric_limits<double>::quiet_NaN());
      
      
      current_statement__ = 14;
      ann = context__.vals_r("ann");
      current_statement__ = 15;
      stan::math::validate_non_negative_index("sigmas", "w", w);
      current_statement__ = 16;
      stan::math::validate_non_negative_index("mu", "t", t);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = w + t;
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "normalx462_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      std::vector<local_scalar_t__> sigmas =
         std::vector<local_scalar_t__>(w, DUMMY_VAR__);
      current_statement__ = 1;
      sigmas = in__.template read_constrain_lb<std::vector<local_scalar_t__>, 
                 jacobian__>(0, lp__, w);
      std::vector<local_scalar_t__> mu =
         std::vector<local_scalar_t__>(t, DUMMY_VAR__);
      current_statement__ = 2;
      mu = in__.template read<std::vector<local_scalar_t__>>(t);
      {
        current_statement__ = 5;
        for (int a_ = 1; a_ <= a; ++a_) {
          current_statement__ = 3;
          lp_accum__.add(
            stan::math::normal_lpdf<propto__>(
              stan::model::rvalue(ann, "ann", stan::model::index_uni(a_)),
              stan::model::rvalue(mu, "mu",
                stan::model::index_uni(stan::model::rvalue(t_A, "t_A",
                                         stan::model::index_uni(a_)))),
              stan::model::rvalue(sigmas, "sigmas",
                stan::model::index_uni(stan::model::rvalue(w_A, "w_A",
                                         stan::model::index_uni(a_))))));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "normalx462_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      std::vector<double> sigmas =
         std::vector<double>(w, std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 1;
      sigmas = in__.template read_constrain_lb<std::vector<local_scalar_t__>, 
                 jacobian__>(0, lp__, w);
      std::vector<double> mu =
         std::vector<double>(t, std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 2;
      mu = in__.template read<std::vector<local_scalar_t__>>(t);
      out__.write(sigmas);
      out__.write(mu);
      if (stan::math::logical_negation((stan::math::primitive_value(
            emit_transformed_parameters__) || stan::math::primitive_value(
            emit_generated_quantities__)))) {
        return ;
      } 
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      } 
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(VecVar& params_r__, VecI& params_i__,
                                   VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      std::vector<local_scalar_t__> sigmas =
         std::vector<local_scalar_t__>(w, DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= w; ++sym1__) {
        sigmas[(sym1__ - 1)] = in__.read<local_scalar_t__>();
      }
      out__.write_free_lb(0, sigmas);
      std::vector<local_scalar_t__> mu =
         std::vector<local_scalar_t__>(t, DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= t; ++sym1__) {
        mu[(sym1__ - 1)] = in__.read<local_scalar_t__>();
      }
      out__.write(mu);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"sigmas", "mu"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{
                                                                   static_cast<size_t>(w)
                                                                   },
      std::vector<size_t>{static_cast<size_t>(t)}};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= w; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "sigmas" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= t; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "mu" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= w; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "sigmas" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= t; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "mu" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"sigmas\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(w) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"mu\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(t) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"sigmas\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(w) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"mu\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(t) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  (w + t);
      const size_t num_transformed = emit_transformed_parameters * 0;
      const size_t num_gen_quantities = emit_generated_quantities * 0;
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      std::vector<int> params_i;
      vars = Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  (w + t);
      const size_t num_transformed = emit_transformed_parameters * 0;
      const size_t num_gen_quantities = emit_generated_quantities * 0;
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      vars = std::vector<double>(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec(params_r.size());
      std::vector<int> params_i;
      transform_inits(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }

  inline void transform_inits(const stan::io::var_context& context,
                              std::vector<int>& params_i,
                              std::vector<double>& vars,
                              std::ostream* pstream__ = nullptr) const {
     constexpr std::array<const char*, 2> names__{"sigmas", "mu"};
      const std::array<Eigen::Index, 2> constrain_param_sizes__{w, t};
      const auto num_constrained_params__ = std::accumulate(
        constrain_param_sizes__.begin(), constrain_param_sizes__.end(), 0);
    
     std::vector<double> params_r_flat__(num_constrained_params__);
     Eigen::Index size_iter__ = 0;
     Eigen::Index flat_iter__ = 0;
     for (auto&& param_name__ : names__) {
       const auto param_vec__ = context.vals_r(param_name__);
       for (Eigen::Index i = 0; i < constrain_param_sizes__[size_iter__]; ++i) {
         params_r_flat__[flat_iter__] = param_vec__[i];
         ++flat_iter__;
       }
       ++size_iter__;
     }
     vars.resize(num_params_r__);
     transform_inits_impl(params_r_flat__, params_i, vars, pstream__);
    } // transform_inits() 
    
};
}
using stan_model = normalx462_model_namespace::normalx462_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return normalx462_model_namespace::profiles__;
}

#endif

