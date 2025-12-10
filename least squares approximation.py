import numpy as np

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def matmul(A, B):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2
    C = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def identity(n):
    I = [[0]*n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I

def inverse(M):
    n = len(M)
    A = [row[:] for row in M]
    I = identity(n)

    for i in range(n):
        pivot = A[i][i]
        if pivot == 0:
            raise ValueError("Matrix not invertible.")

        # Normalize pivot row
        for j in range(n):
            A[i][j] /= pivot
            I[i][j] /= pivot

        # Eliminate all other rows
        for r in range(n):
            if r != i:
                factor = A[r][i]
                for c in range(n):
                    A[r][c] -= factor * A[i][c]
                    I[r][c] -= factor * I[i][c]

    return I

class LeastSquaresElementary:
    def fit(self, X, y):
        X = X.tolist()
        y = y.reshape(-1,1).tolist()

        # store raw data for summary()
        self.X_raw = X
        self.y_raw = y

        # Add intercept
        X_aug = [[1] + row for row in X]
        self.X_aug = X_aug

        Xt = transpose(X_aug)
        XtX = matmul(Xt, X_aug)
        XtX_inv = inverse(XtX)
        Xty = matmul(Xt, y)

        self.beta = matmul(XtX_inv, Xty)
        return self

    def predict(self, X):
        X = X.tolist()
        X_aug = [[1] + row for row in X]
        y_pred = matmul(X_aug, self.beta)
        return np.array(y_pred).flatten()

    def summary(self):
        """
        Print an elementary regression summary using only manual computations.
        """
        # Convert back to arrays for simple loop math
        X = self.X_aug
        y = np.array(self.y_raw).flatten()

        # Predicted values
        y_pred = self.predict(np.array(self.X_raw))

        # Residuals
        residuals = y - y_pred

        # Residual variance σ² = RSS / (n - p)
        RSS = sum(residuals**2)
        n = len(y)
        p = len(self.beta)  # includes intercept
        sigma2 = RSS / (n - p)

        # Total Sum of Squares
        y_mean = sum(y)/n
        TSS = sum((y - y_mean)**2)

        # R^2
        R2 = 1 - RSS/TSS

        # --- Print summary ---
        print("============== LEAST SQUARES SUMMARY ==============")
        print("Coefficients:")
        for i, b in enumerate(self.beta):
            name = "Intercept" if i == 0 else f"Beta {i}"
            print(f"  {name}: {b[0]:.4f}")
        print("---------------------------------------------------")
        print(f"Residual Sum of Squares (RSS):  {RSS:.4f}")
        print(f"Residual Variance (sigma^2):    {sigma2:.4f}")
        print(f"R^2:                            {R2:.4f}")
        print("===================================================")
