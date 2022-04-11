using OptimizationProblems, NLPModelsJuMP
using NLPModels
using LinearAlgebra
using Krylov
using BenchmarkTools
using LinearOperators


modele_jump = OptimizationProblems.arwhead(500)
nlpmodel = MathOptNLPModel(modele_jump, name="arwhead ")

function trust_region(nlpmodel)
    n = nlpmodel.meta.nvar
    x = rand(n)
    gx = NLPModels.grad(nlpmodel, x)
    g0 = gx
    B = LBFGSOperator(n)
    Δ0 = (Float64)(1)
    Δmax = 10
    Δ = Δ0
    η = 1e-3 :: Float64

    while norm(gx) > norm(g0) * 10e-6
        fx = NLPModels.obj(nlpmodel, x) 
        s = calcul_s(gx, B, Δ)
        p = calcul_p(x, s, fx, gx, B, nlpmodel)
        if  p < 0.25
            Δ = Δ/ 4
        elseif p > 0.75 && norm(s) == Δ
            Δ = min(2 * Δ, Δmax) 
        else
            Δ = Δ
        end
        if p ≥ η
            x = x + s
            gprec = gx
            gx = NLPModels.grad(nlpmodel, x)
            y = gx - gprec
            push!(B,s,y)
        else
            x = x
        end
      
        println("norme gx = ", norm(gx)," fx = ", fx," Delta = ", Δ, " p = ", p, " norme s = ", norm(s))
    end

end

function calcul_s(gx, B, Δ)
    (s, stats) = cg(B,-gx, radius = Δ)
    return s
end

function calcul_p(x, s, fx, gx, B, nlpmodel)
    fxs = NLPModels.obj(nlpmodel, x .+ s)
    msk = fx + dot(gx, s) + (1/2) * dot(s, B*s) 
 
    p = (fx - fxs)/(fx - msk)
    return p
end 

