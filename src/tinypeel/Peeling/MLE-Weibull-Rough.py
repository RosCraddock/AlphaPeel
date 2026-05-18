#-----Test 1 of applying a gradient descent approach to estimate the three parameters for the Weibull model.-------
# This gave a mean accuracy of 0.4219
# def _project_params(p):
#     p = np.array(p, dtype=np.float64)
#     p[0] = np.clip(p[0], 1e-8, 1.0 - 1e-8)  # k
#     p[1] = max(p[1], 1e-8)                  # lam
#     p[2] = max(p[2], 1e-8)                  # alpha
#     return p

# def negative_log_likelihood(params, pheno, age, genoProb):
#     k, lam, alpha = params
#     eps = 1e-8
#     f_t = (1-k)*(1-np.exp(-lam*age**alpha))
#     f_t = np.clip(f_t, eps, 1-eps)  # f_t must be between 0 and 1
#     genoProb = np.clip(genoProb, eps, 1-eps)  # Geno probabilities must be between 0 and 1
#     log_likelihood = np.sum(pheno*np.log(f_t) + (1-pheno)*np.log(1-f_t))
#     log_likelihood += np.sum(np.log(genoProb))
#     return -log_likelihood
def negative_log_likelihood(params, pheno, age, genoProb, q_0=1e-4):
    k, lam, alpha = params
    eps = 1e-8
    f_t = (1-k)*(1-np.exp(-lam*age**alpha))
    f_t = np.clip(f_t, eps, 1-eps)  # f_t must be between 0 and 1
    genoProb = np.clip(genoProb, eps, 1-eps)  # Geno probabilities must be between 0 and 1
    p_case = genoProb*f_t + (1-genoProb)*q_0 # Probability of being a case given genotype probabilities and penetrance
    log_likelihood = np.sum(pheno*np.log(p_case) + (1-pheno)*np.log(1-p_case))
    return -log_likelihood

def _fit_weibull_gd(pheno, age, genoProb, init_params, learning_rate=0.01, max_iters=100, min_iters=10, tol=1e-8, grad_tol=1e-8, patience=5):
    """Gradient descent with early stopping."""
    params = np.array(init_params, dtype=np.float64)
    best_params = params.copy()
    best_loss = negative_log_likelihood(params, pheno, age, genoProb)
    no_improve = 0
    delta_inf = 1e-1
    delta = 9e-2

    for iteration in range(max_iters):
        print(f"Iteration {iteration+1}/{max_iters}, Loss: {best_loss:.6f}, Params: {params}")  
        # Numerical gradient (central difference is usually more stable than forward difference)
        grad = np.zeros(3, dtype=np.float64)
        for i in range(3):
            p_plus = params.copy()
            p_minus = params.copy()
            if i == 0: #k
                p_plus[i] += delta
                p_minus[i] -= delta
            else: #lam or alpha
                p_plus[i] += delta_inf
                p_minus[i] -= delta_inf
            loss_plus = negative_log_likelihood(p_plus, pheno, age, genoProb)
            loss_minus = negative_log_likelihood(p_minus, pheno, age, genoProb)
            if i == 0: 
                grad[i] = (loss_plus - loss_minus) / (2.0 * delta)
            else:
                grad[i] = (loss_plus - loss_minus) / (2.0 * delta_inf)
            

            # p_plus = _project_params(params.copy())
            # p_minus = _project_params(params.copy())
            # p_plus[i] += delta
            # p_minus[i] -= delta
            # p_plus = _project_params(p_plus)
            # p_minus = _project_params(p_minus)

            # loss_plus = negative_log_likelihood(p_plus, pheno, age, genoProb)
            # loss_minus = negative_log_likelihood(p_minus, pheno, age, genoProb)
            # grad[i] = (loss_plus - loss_minus) / (2.0 * delta)

        # Parameter update
        params[0] -= learning_rate * grad[0]
        params[1] -= learning_rate * grad[1]
        params[2] -= learning_rate * grad[2]

        # Bounds
        params[0] = np.clip(params[0], 1e-8, 1.0 - 1e-8)
        params[1] = np.clip(params[1], 1e-8, None)
        params[2] = np.clip(params[2], 1e-8, None)

        loss = negative_log_likelihood(params, pheno, age, genoProb)

        # Track best
        if loss + tol < best_loss:
            best_loss = loss
            best_params = params.copy()
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if iteration + 1 >= min_iters:
            if np.linalg.norm(grad) < grad_tol or no_improve >= patience:
                break

    return best_params[0], best_params[1], best_params[2]

def updateIndPhenoPenetrance(pedigree, peelingInfo):
   # TODO: Implement this function, initially for my scenario (ignore complexity of how user will inform etc).
   # This will use an optimisation approach (i.e MLE) to estimate the three parameters in an extended Weibull model.
    pheno_all = []
    age_all = []
    genoProb_all = []

    for ind in pedigree:
        if ind.phenotype is None:
            continue

        pheno = [int(np.asarray(p).reshape(-1)[0]) for p in ind.phenotype]
        age = [float(np.asarray(a).reshape(-1)[0]) for a in ind.age] if ind.age is not None else [0.0] * len(pheno)
        p_g3 = [float(peelingInfo.getGenoProbs(ind.idn)[3, 0])] * len(pheno)

        pheno_all.extend(pheno)
        age_all.extend(age)
        genoProb_all.extend(p_g3)
    
        

    if len(pheno_all) == 0:
        print("No phenotype records found; skipping update.")
        return
    
    pheno_all = np.asarray(pheno_all, dtype=np.float64)
    age_all = np.asarray(age_all, dtype=np.float64)
    genoProb_all = np.asarray(genoProb_all, dtype=np.float64)

    # Remove all with an age of 0 (assume an adjusted age, where 0 is equivalent to the max age not to be diseased).
    keep = age_all != 0.0
    pheno_all = pheno_all[keep]
    age_all = age_all[keep]
    genoProb_all = genoProb_all[keep]

    #results = minimize(negative_log_likelihood, pedigree.weibullParams, args=(pheno_all, age_all, genoProb_all), method = 'L-BFGS-B', bounds = [(1e-8, 1-1e-8), (1e-8, None), (1e-8, None)])

    #if not results.success:
    #    print("Weibull optimisation failed:", results.message)
    #    return
    
    #pedigree.weibullParams = results.x.astype(np.float32)

    # Replace the minimize call with:
    k, lam, alpha = _fit_weibull_gd(
        pheno_all, age_all, genoProb_all, 
        pedigree.weibullParams, 
        learning_rate=0.01/len(pheno_all)**0.5, 
        max_iters=100
    )
    pedigree.weibullParams = np.array([k, lam, alpha], dtype=np.float32)
    print("Updated individual phenotype penetrance parameters:", pedigree.weibullParams)

    # Update the individual phenotype penetrance matrix based on the new parameters.
    # For now, will only update the two columns of the fourth row
    for ind in pedigree:
        if ind.phenotype is not None:
            for r in range(len(ind.phenotype)):
                age = float(np.asarray(ind.age[r]).reshape(-1)[0])
                f_t = (1-pedigree.weibullParams[0])*(1-np.exp(-pedigree.weibullParams[1]*age**pedigree.weibullParams[2]))
                f_t = np.float32(np.clip(f_t, 1e-8, 1-1e-8))  # f_t must be between 0 and 1
                ind.indPhenoPenetrance[r][3, 0] = 1-f_t
                ind.indPhenoPenetrance[r][3, 1] = f_t

#-----Test 2 of applying a Newton Raphson approach to estimate the three parameters for the Weibull model.-------
# This gave a mean accuracy of 0.4333
def _clip_params(k, lam, alpha, k_lo=1e-8, k_hi=1-1e-8, lo=1e-8):
    k = float(np.clip(k, k_lo, k_hi))
    lam = float(max(lam, lo))
    alpha = float(max(alpha, lo))
    return k, lam, alpha

def _model_terms(k, lam, alpha, y, g, age, q0=1e-4, eps=1e-10):
    # y: phenotype 0/1
    # g: P(aa)
    # age must be > 0 for alpha derivatives (log age)
    y = np.asarray(y, dtype=np.float64)
    g = np.clip(np.asarray(g, dtype=np.float64), eps, 1.0 - eps)
    age = np.asarray(age, dtype=np.float64)

    # enforce strictly positive age for alpha derivative stability
    age = np.maximum(age, 1e-8)

    A = np.power(age, alpha)
    E = np.exp(-lam * A)

    # p_case = b + a, where a depends on k,lam,alpha
    C = g * (1.0 - k)
    a = C * (1.0 - E)
    b = (1.0 - g) * q0
    p = b + a
    p = np.clip(p, eps, 1.0 - eps)

    return y, g, age, A, E, p

def negative_log_likelihood(params, y, g, age, q0=1e-4):
    k, lam, alpha = params
    k, lam, alpha = _clip_params(k, lam, alpha)
    y, g, age, A, E, p = _model_terms(k, lam, alpha, y, g, age, q0=q0)
    return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

def _grad_hess_lam_alpha(k, lam, alpha, y, g, age, q0=1e-4, eps=1e-10):
    y, g, age, A, E, p = _model_terms(k, lam, alpha, y, g, age, q0=q0, eps=eps)

    C = g * (1.0 - k)
    L = np.log(age)

    # first derivatives of p
    dp_l = C * A * E
    dp_a = C * lam * A * L * E

    # second derivatives of p
    d2p_ll = -C * (A ** 2) * E
    d2p_aa = C * lam * A * (L ** 2) * E * (1.0 - lam * A)
    d2p_la = C * A * L * E * (1.0 - lam * A)

    # nll weights
    w = (1.0 - y) / (1.0 - p) - y / p
    v = y / (p ** 2) + (1.0 - y) / ((1.0 - p) ** 2)

    # gradient of nll
    g_l = np.sum(w * dp_l)
    g_a = np.sum(w * dp_a)

    # Hessian of nll (2x2)
    H_ll = np.sum(v * dp_l * dp_l + w * d2p_ll)
    H_aa = np.sum(v * dp_a * dp_a + w * d2p_aa)
    H_la = np.sum(v * dp_l * dp_a + w * d2p_la)

    grad = np.array([g_l, g_a], dtype=np.float64)
    H = np.array([[H_ll, H_la],
                  [H_la, H_aa]], dtype=np.float64)
    return grad, H

def _update_k_by_bisection(k, lam, alpha, y, g, age, q0=1e-4, k_lo=1e-8, k_hi=1-1e-8, max_iter=60):
    # Solve grad_k(nll)=0 by bisection. No Hessian used for k.
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    age = np.maximum(np.asarray(age, dtype=np.float64), 1e-8)

    A = np.power(age, alpha)
    E = np.exp(-lam * A)

    # dp/dk = -g*(1-E), independent of k
    dp_dk = -g * (1.0 - E)

    def grad_k(kv):
        p = g * (1.0 - kv) * (1.0 - E) + (1.0 - g) * q0
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        w = (1.0 - y) / (1.0 - p) - y / p
        return np.sum(w * dp_dk)

    glo = grad_k(k_lo)
    ghi = grad_k(k_hi)

    # If no root in bracket, optimum at boundary
    if glo >= 0.0:
        return k_lo
    if ghi <= 0.0:
        return k_hi

    lo, hi = k_lo, k_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gm = grad_k(mid)
        if gm > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def fit_weibull_weighted(y, g, age, init_params, q0=1e-4, max_iter=100, tol=1e-8):
    k, lam, alpha = _clip_params(*init_params)
    best = np.array([k, lam, alpha], dtype=np.float64)
    best_nll = negative_log_likelihood(best, y, g, age, q0=q0)

    for it in range(max_iter):
        # Step 1: lambda/alpha Newton with damping
        grad, H = _grad_hess_lam_alpha(k, lam, alpha, y, g, age, q0=q0)
        H_reg = H + 1e-6 * np.eye(2)

        try:
            step = np.linalg.solve(H_reg, grad)
        except np.linalg.LinAlgError:
            step = np.zeros(2, dtype=np.float64)

        # backtracking line search
        old_nll = negative_log_likelihood([k, lam, alpha], y, g, age, q0=q0)
        t = 1.0
        accepted = False
        for _ in range(20):
            lam_c = max(lam - t * step[0], 1e-8)
            alpha_c = max(alpha - t * step[1], 1e-8)
            nll_c = negative_log_likelihood([k, lam_c, alpha_c], y, g, age, q0=q0)
            if np.isfinite(nll_c) and nll_c <= old_nll:
                lam, alpha = lam_c, alpha_c
                accepted = True
                break
            t *= 0.5
        if not accepted:
            lam = max(lam - 1e-3 * grad[0], 1e-8)
            alpha = max(alpha - 1e-3 * grad[1], 1e-8)

        # Step 2: k update by gradient root (bisection), no Hessian
        k = _update_k_by_bisection(k, lam, alpha, y, g, age, q0=q0)

        # track best
        cur = np.array([k, lam, alpha], dtype=np.float64)
        cur_nll = negative_log_likelihood(cur, y, g, age, q0=q0)
        if cur_nll < best_nll:
            best_nll = cur_nll
            best = cur.copy()

        # convergence
        if np.max(np.abs(cur - best)) < tol:
            break

    return best

#-----End of test 2.-------
#-----Third test
# This gave a mean accuracy of 0.4490
def _clip_params(k, lam, alpha, k_lo=1e-8, k_hi=1.0 - 1e-8, lo=1e-8):
    k = float(np.clip(k, k_lo, k_hi))
    lam = float(max(lam, lo))
    alpha = float(max(alpha, lo))
    return k, lam, alpha


def _p_case_terms(k, lam, alpha, g, age, e, eps=1e-10):
    g = np.clip(np.asarray(g, dtype=np.float64), eps, 1.0 - eps)
    age = np.maximum(np.asarray(age, dtype=np.float64), 1e-8)  # for log(age)

    A = np.power(age, alpha)
    E = np.exp(-lam * A)

    # p_case = P(Y=1) = g*(1-k)*(1-exp(-lam*age^alpha)) + (1-g)*e
    p_case = g * (1.0 - k) * (1.0 - E) + (1.0 - g) * e
    p_case = np.clip(p_case, eps, 1.0 - eps)
    return p_case, A, E, g, age


def _nll(k, lam, alpha, y, g, age, e):
    p_case, _, _, _, _ = _p_case_terms(k, lam, alpha, g, age, e)
    y = np.asarray(y, dtype=np.float64)
    return -np.sum(y * np.log(p_case) + (1.0 - y) * np.log(1.0 - p_case))


def _grad_hess_lam_alpha(k, lam, alpha, y, g, age, e):
    y = np.asarray(y, dtype=np.float64)
    p, A, E, g, age = _p_case_terms(k, lam, alpha, g, age, e)
    C = g * (1.0 - k)
    L = np.log(age)

    # First derivatives of p_case
    dp_l = C * A * E
    dp_a = C * lam * A * L * E

    # Second derivatives of p_case
    d2p_ll = -C * (A ** 2) * E
    d2p_aa = C * lam * A * (L ** 2) * E * (1.0 - lam * A)
    d2p_la = C * A * L * E * (1.0 - lam * A)

    # NLL derivatives
    w = (1.0 - y) / (1.0 - p) - y / p
    v = y / (p ** 2) + (1.0 - y) / ((1.0 - p) ** 2)

    g_l = np.sum(w * dp_l)
    g_a = np.sum(w * dp_a)

    H_ll = np.sum(v * dp_l * dp_l + w * d2p_ll)
    H_aa = np.sum(v * dp_a * dp_a + w * d2p_aa)
    H_la = np.sum(v * dp_l * dp_a + w * d2p_la)

    grad = np.array([g_l, g_a], dtype=np.float64)
    H = np.array([[H_ll, H_la], [H_la, H_aa]], dtype=np.float64)
    return grad, H


def _update_k_bisection(lam, alpha, y, g, age, e, k_lo=1e-8, k_hi=1.0 - 1e-8, max_iter=60):
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    age = np.maximum(np.asarray(age, dtype=np.float64), 1e-8)

    A = np.power(age, alpha)
    E = np.exp(-lam * A)
    dp_dk = -g * (1.0 - E)  # exact derivative of p_case wrt k

    def grad_k(kv):
        p = g * (1.0 - kv) * (1.0 - E) + (1.0 - g) * e
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        w = (1.0 - y) / (1.0 - p) - y / p
        return np.sum(w * dp_dk)

    glo = grad_k(k_lo)
    ghi = grad_k(k_hi)

    # If no root in bracket, optimum is at boundary
    if glo >= 0.0:
        return k_lo
    if ghi <= 0.0:
        return k_hi

    lo, hi = k_lo, k_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gm = grad_k(mid)
        if gm > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def iter_ind_pheno_penetrance(weibullParams, phenotypes, genoProb, adjustedAge, e=0.01):
    k, lam, alpha = _clip_params(*weibullParams)

    # 1) Newton step for lam/alpha with damping
    grad, H = _grad_hess_lam_alpha(k, lam, alpha, phenotypes, genoProb, adjustedAge, e)
    H = H + 1e-6 * np.eye(2)

    try:
        step = np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        step = np.zeros(2, dtype=np.float64)

    old_nll = _nll(k, lam, alpha, phenotypes, genoProb, adjustedAge, e)
    t = 1.0
    for _ in range(20):
        lam_c = max(lam - t * step[0], 1e-8)
        alpha_c = max(alpha - t * step[1], 1e-8)
        new_nll = _nll(k, lam_c, alpha_c, phenotypes, genoProb, adjustedAge, e)
        if np.isfinite(new_nll) and new_nll <= old_nll:
            lam, alpha = lam_c, alpha_c
            break
        t *= 0.5

    # 2) k update by score root (no Hessian for k)
    k = _update_k_bisection(lam, alpha, phenotypes, genoProb, adjustedAge, e)

    return k, lam, alpha

def updateIndPhenoPenetrance(pedigree, peelingInfo, e=0.01, maxIter=1000, tol=0.01):
    pheno_all = []
    age_all = []
    genoProb_all = []

    for ind in pedigree:
        if ind.phenotype is None:
            continue

        pheno = [int(np.asarray(p).reshape(-1)[0]) for p in ind.phenotype]
        age = [float(np.asarray(a).reshape(-1)[0]) for a in ind.age] if ind.age is not None else [0.0] * len(pheno)
        p_g3 = [float(peelingInfo.getGenoProbs(ind.idn)[3, 0])] * len(pheno)

        pheno_all.extend(pheno)
        age_all.extend(age)
        genoProb_all.extend(p_g3)
    
        

    if len(pheno_all) == 0:
        print("No phenotype records found; skipping update.")
        return
    
    pheno_all = np.asarray(pheno_all, dtype=np.float64)
    age_all = np.maximum(np.asarray(age_all, dtype=np.float64), 1e-8)
    genoProb_all = np.asarray(genoProb_all, dtype=np.float64)
    
    for iters in range(maxIter):
        k, lambda_, alpha = pedigree.weibullParams
        k_new, lambda_new, alpha_new = iter_ind_pheno_penetrance(
    pedigree.weibullParams, pheno_all, genoProb_all, age_all, e=e
    )

        if (np.isclose(k, k_new, atol=tol) and 
            np.isclose(lambda_, lambda_new, atol=tol) and 
            np.isclose(alpha, alpha_new, atol=tol)):
            print("Exiting on iteration: ", iters)
            break

        pedigree.weibullParams = [k_new, lambda_new, alpha_new]

    print("Updated individual phenotype penetrance parameters:", pedigree.weibullParams)

    # Update the individual phenotype penetrance matrix based on the new parameters.
    # For now, will only update the two columns of the fourth row
    for ind in pedigree:
        if ind.phenotype is not None:
            for r in range(len(ind.phenotype)):
                age = float(np.asarray(ind.age[r]).reshape(-1)[0])
                f_t = (1.0 - pedigree.weibullParams[0]) * (1.0 - np.exp(-pedigree.weibullParams[1] * np.power(age, pedigree.weibullParams[2])))
                f_t = np.float32(np.clip(f_t, 1e-8, 1-1e-8))  # f_t must be between 0 and 1
                ind.indPhenoPenetrance[r][3, 0] = 1-f_t
                ind.indPhenoPenetrance[r][3, 1] = f_t

#-----Fourth test
# Newton optimisation for all three parameters: Performed the best with a mean accuracy of 0.4818
# This gave a mean accuracy of 0.4490
def _clip_params(k, lam, alpha, k_lo=1e-8, k_hi=1.0 - 1e-8, lo=1e-8):
    k = float(np.clip(k, k_lo, k_hi))
    lam = float(max(lam, lo))
    alpha = float(max(alpha, lo))
    return k, lam, alpha


def _p_case_terms(k, lam, alpha, g, age, e, eps=1e-10):
    g = np.clip(np.asarray(g, dtype=np.float64), eps, 1.0 - eps)
    age = np.maximum(np.asarray(age, dtype=np.float64), 1e-8)  # for log(age)

    A = np.power(age, alpha)
    E = np.exp(-lam * A)

    # p_case = P(Y=1) = g*(1-k)*(1-exp(-lam*age^alpha)) + (1-g)*e
    p_case = g * (1.0 - k) * (1.0 - E) + (1.0 - g) * e
    p_case = np.clip(p_case, eps, 1.0 - eps)
    return p_case, A, E, g, age


def _nll(k, lam, alpha, y, g, age, e):
    p_case, _, _, _, _ = _p_case_terms(k, lam, alpha, g, age, e)
    y = np.asarray(y, dtype=np.float64)
    return -np.sum(y * np.log(p_case) + (1.0 - y) * np.log(1.0 - p_case))


def _grad_hess_lam_alpha(k, lam, alpha, y, g, age, e):
    y = np.asarray(y, dtype=np.float64)
    p, A, E, g, age = _p_case_terms(k, lam, alpha, g, age, e)
    C = g * (1.0 - k)
    L = np.log(age)

    # First derivatives of p_case
    dp_l = C * A * E
    dp_a = C * lam * A * L * E

    # Second derivatives of p_case
    d2p_ll = -C * (A ** 2) * E
    d2p_aa = C * lam * A * (L ** 2) * E * (1.0 - lam * A)
    d2p_la = C * A * L * E * (1.0 - lam * A)

    # NLL derivatives
    w = (1.0 - y) / (1.0 - p) - y / p
    v = y / (p ** 2) + (1.0 - y) / ((1.0 - p) ** 2)

    g_l = np.sum(w * dp_l)
    g_a = np.sum(w * dp_a)

    H_ll = np.sum(v * dp_l * dp_l + w * d2p_ll)
    H_aa = np.sum(v * dp_a * dp_a + w * d2p_aa)
    H_la = np.sum(v * dp_l * dp_a + w * d2p_la)

    grad = np.array([g_l, g_a], dtype=np.float64)
    H = np.array([[H_ll, H_la], [H_la, H_aa]], dtype=np.float64)
    return grad, H


def _grad_hess_k(k, lam, alpha, y, g, age, e):
    y = np.asarray(y, dtype=np.float64)
    p, A, E, g, age = _p_case_terms(k, lam, alpha, g, age, e)

    # dp/dk = -g*(1-E), d2p/dk2 = 0
    dp_k = -g * (1.0 - E)

    # NLL derivatives
    w = (1.0 - y) / (1.0 - p) - y / p
    v = y / (p ** 2) + (1.0 - y) / ((1.0 - p) ** 2)

    grad_k = np.sum(w * dp_k)
    hess_k = np.sum(v * dp_k * dp_k) + 1e-12  # ridge for stability
    return grad_k, hess_k


def _update_k_newton(k, lam, alpha, y, g, age, e, k_lo=1e-8, k_hi=1.0 - 1e-8, max_ls=20):
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    age = np.maximum(np.asarray(age, dtype=np.float64), 1e-8)

    grad_k, hess_k = _grad_hess_k(k, lam, alpha, y, g, age, e)
    step = grad_k / hess_k

    old_nll = _nll(k, lam, alpha, y, g, age, e)
    t = 1.0
    for _ in range(max_ls):
        k_c = float(np.clip(k - t * step, k_lo, k_hi))
        new_nll = _nll(k_c, lam, alpha, y, g, age, e)
        if np.isfinite(new_nll) and new_nll <= old_nll:
            return k_c
        t *= 0.5

    return k


def iter_ind_pheno_penetrance(weibullParams, phenotypes, genoProb, adjustedAge, e=0.01):
    k, lam, alpha = _clip_params(*weibullParams)

    # 1) Newton step for lam/alpha with damping
    grad, H = _grad_hess_lam_alpha(k, lam, alpha, phenotypes, genoProb, adjustedAge, e)
    H = H + 1e-6 * np.eye(2)

    try:
        step = np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        step = np.zeros(2, dtype=np.float64)

    old_nll = _nll(k, lam, alpha, phenotypes, genoProb, adjustedAge, e)
    t = 1.0
    for _ in range(20):
        lam_c = max(lam - t * step[0], 1e-8)
        alpha_c = max(alpha - t * step[1], 1e-8)
        new_nll = _nll(k, lam_c, alpha_c, phenotypes, genoProb, adjustedAge, e)
        if np.isfinite(new_nll) and new_nll <= old_nll:
            lam, alpha = lam_c, alpha_c
            break
        t *= 0.5

    # 2) k update by newton
    k = _update_k_newton(k, lam, alpha, phenotypes, genoProb, adjustedAge, e)

    return k, lam, alpha

def updateIndPhenoPenetrance(pedigree, peelingInfo, e=0.01, maxIter=1000, tol=0.01):
    pheno_all = []
    age_all = []
    genoProb_all = []

    for ind in pedigree:
        if ind.phenotype is None:
            continue

        pheno = [int(np.asarray(p).reshape(-1)[0]) for p in ind.phenotype]
        age = [float(np.asarray(a).reshape(-1)[0]) for a in ind.age] if ind.age is not None else [0.0] * len(pheno)
        p_g3 = [float(peelingInfo.getGenoProbs(ind.idn)[3, 0])] * len(pheno)

        pheno_all.extend(pheno)
        age_all.extend(age)
        genoProb_all.extend(p_g3)
    
        

    if len(pheno_all) == 0:
        print("No phenotype records found; skipping update.")
        return
    
    pheno_all = np.asarray(pheno_all, dtype=np.float64)
    age_all = np.maximum(np.asarray(age_all, dtype=np.float64), 1e-8)
    genoProb_all = np.asarray(genoProb_all, dtype=np.float64)
    
    for iters in range(maxIter):
        k, lambda_, alpha = pedigree.weibullParams
        k_new, lambda_new, alpha_new = iter_ind_pheno_penetrance(
    pedigree.weibullParams, pheno_all, genoProb_all, age_all, e=e
    )

        if (np.isclose(k, k_new, atol=tol) and 
            np.isclose(lambda_, lambda_new, atol=tol) and 
            np.isclose(alpha, alpha_new, atol=tol)):
            print("Exiting on iteration: ", iters)
            break

        pedigree.weibullParams = [k_new, lambda_new, alpha_new]

    print("Updated individual phenotype penetrance parameters:", pedigree.weibullParams)

    # Update the individual phenotype penetrance matrix based on the new parameters.
    # For now, will only update the two columns of the fourth row
    for ind in pedigree:
        if ind.phenotype is not None:
            for r in range(len(ind.phenotype)):
                age = float(np.asarray(ind.age[r]).reshape(-1)[0])
                f_t = (1.0 - pedigree.weibullParams[0]) * (1.0 - np.exp(-pedigree.weibullParams[1] * np.power(age, pedigree.weibullParams[2])))
                f_t = np.float32(np.clip(f_t, 1e-8, 1-1e-8))  # f_t must be between 0 and 1
                ind.indPhenoPenetrance[r][3, 0] = 1 - f_t
                ind.indPhenoPenetrance[r][3, 1] = f_t



# Test using scipy.optimise minimize() function
# Test with estimation of e as well? & update the other penetrance values accordingly?
# This gave the highest accuracy across all models with 0.6521
# Note that the parameters (k, lambda, alpha, and e) are all transformed for optimisation. The idea was that it would allow the parameters to be the same scale.
# However, this made no difference to the accuracy (measured as the Pearson correlation between estimated allele dosage and true).
def negativeLogLikelihoodPhenoPenetrance(params, pheno_all, genoProb_all, age_all):
    k, lambda_, alpha, e = params
    k = 1 / (1 + np.exp(-k))  # inverse logit transform for k
    lambda_ = np.exp(lambda_)  # log transform for lambda
    alpha = np.exp(alpha)  # log transform for alpha
    e = 1 / (1 + np.exp(-e))  # inverse logit transform for e
    nll = 0.0
    p_case = genoProb_all*(1.0-k)*(1.0-np.exp(-lambda_ * np.power(age_all, alpha)))+e*(1.0-genoProb_all)
    p_case = np.clip(p_case, 1e-10, 1 - 1e-10)  # Prevent log(0) or log(1)
    nll = -np.sum(pheno_all * np.log(p_case) + (1.0 - pheno_all) * np.log(1.0 - p_case))
    return nll

def updateIndPhenoPenetrance(pedigree, peelingInfo, maxIter=1000, tol=0.01):
    pheno_all = []
    age_all = []
    genoProb_all = []

    for ind in pedigree:
        if ind.phenotype is None:
            continue

        pheno = [int(np.asarray(p).reshape(-1)[0]) for p in ind.phenotype]
        age = [float(np.asarray(a).reshape(-1)[0]) for a in ind.age] if ind.age is not None else [0.0] * len(pheno)
        p_g3 = [float(peelingInfo.getGenoProbs(ind.idn)[3, 0])] * len(pheno)

        pheno_all.extend(pheno)
        age_all.extend(age)
        genoProb_all.extend(p_g3)
    
        

    if len(pheno_all) == 0:
        print("No phenotype records found; skipping update.")
        return
    
    pheno_all = np.asarray(pheno_all, dtype=np.float64)
    age_all = np.maximum(np.asarray(age_all, dtype=np.float64), 1e-8)
    genoProb_all = np.asarray(genoProb_all, dtype=np.float64)

    constrained_params = pedigree.weibullParams.copy()
    constrained_params[0] = np.clip(constrained_params[0], 1e-8, 1 - 1e-8)  # k
    constrained_params[1] = max(constrained_params[1], 1e-8)  # lambda
    constrained_params[2] = max(constrained_params[2], 1e-8)  # alpha
    constrained_params[3] = np.clip(constrained_params[3], 1e-8, 1 - 1e-8)  # e

    constrained_params[0] =np.log(constrained_params[0] / (1 - constrained_params[0]))  # logit transform for k
    constrained_params[1] = np.log(constrained_params[1])  # log transform for lambda
    constrained_params[2] = np.log(constrained_params[2])  # log transform for alpha
    constrained_params[3] = np.log(constrained_params[3] / (1 - constrained_params[3]))  # logit transform for e
    
    results = minimize(
        negativeLogLikelihoodPhenoPenetrance,
        constrained_params,
        args=(pheno_all, genoProb_all, age_all),
        method='L-BFGS-B',
        bounds=[(-8, 8), (-12, None), (-12, None), (-8, 8)])

    constrained_params = results.x
    constrained_params[0] = 1 / (1 + np.exp(-constrained_params[0]))  # inverse logit transform for k
    constrained_params[1] = np.exp(constrained_params[1])  # log transform
    constrained_params[2] = np.exp(constrained_params[2])  # log transform
    constrained_params[3] = 1 / (1 + np.exp(-constrained_params[3]))  # inverse logit transform for e
    pedigree.weibullParams = constrained_params.astype(np.float32)

    print("Updated individual phenotype penetrance parameters:", pedigree.weibullParams)

    # Update the individual phenotype penetrance matrix based on the new parameters.
    # For now, will only update the two columns of the fourth row
    for ind in pedigree:
        if ind.phenotype is not None:
            for r in range(len(ind.phenotype)):
                age = float(np.asarray(ind.age[r]).reshape(-1)[0])
                f_t = (1.0 - pedigree.weibullParams[0]) * (1.0 - np.exp(-pedigree.weibullParams[1] * np.power(age, pedigree.weibullParams[2])))
                f_t = np.float32(np.clip(f_t, 1e-8, 1-1e-8))  # f_t must be between 0 and 1
                ind.indPhenoPenetrance[r][3, 0] = 1-f_t # P(unaffected | g=3) = 1 - f(t)
                ind.indPhenoPenetrance[r][3, 1] = f_t # P(affected | g=3) = f(t)
                ind.indPhenoPenetrance[r][0, 0] = 1 -pedigree.weibullParams[3]
                ind.indPhenoPenetrance[r][0, 1] = pedigree.weibullParams[3]
                ind.indPhenoPenetrance[r][1, 0] = 1 -pedigree.weibullParams[3]
                ind.indPhenoPenetrance[r][1, 1] = pedigree.weibullParams[3]
                ind.indPhenoPenetrance[r][2, 0] = 1 -pedigree.weibullParams[3]
                ind.indPhenoPenetrance[r][2, 1] = pedigree.weibullParams[3]


# Genotype probabilities will partially capture pedigree structure, but not explicitly, hence the iid assumption likely remains.
# A suggested fix is to use a mixed model with random effect for family
