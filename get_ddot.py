import numpy as np

def calculate_accelerations(theta, phi, theta_dot, phi_dot, m_c, m_p, l, g, F):
    cos_theta = np.cos(theta)
    cos_theta_phi = np.cos(theta + phi)
    sin_theta = np.sin(theta)
    sin_theta_phi = np.sin(theta + phi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    M = np.zeros((3, 3))
    
    M[0, 0] = m_c + 2*m_p
    M[0, 1] = (3/2)*m_p*l*cos_theta + (1/2)*m_p*l*cos_theta_phi
    M[0, 2] = (1/2)*m_p*l*cos_theta_phi

    M[1, 0] = (3/2)*m_p*l*cos_theta + (1/2)*m_p*l*cos_theta_phi
    M[1, 1] = (5/3)*m_p*l**2 + (1/2)*m_p*l**2*cos_phi
    M[1, 2] = (1/2)*m_p*l**2*cos_phi + (1/3)*m_p*l**2

    M[2, 0] = (1/2)*m_p*l*cos_theta_phi
    M[2, 1] = (1/2)*m_p*l**2*cos_phi + (1/3)*m_p*l**2
    M[2, 2] = (1/3)*m_p*l**2

    b = np.zeros(3)

    b[0] = F + (3/2)*m_p*l*theta_dot**2*sin_theta + (1/2)*m_p*l*(theta_dot + phi_dot)**2*sin_theta_phi

    b[1] = (1/2)*m_p*l**2*(theta_dot + phi_dot)*phi_dot*sin_phi + \
           (3/2)*m_p*g*l*sin_theta + \
           (1/2)*m_p*g*l*sin_theta_phi

    b[2] = (1/2)*m_p*g*l*sin_theta_phi - \
           (1/2)*m_p*l**2*theta_dot**2*sin_phi

    accelerations = np.linalg.solve(M, b)

    return accelerations