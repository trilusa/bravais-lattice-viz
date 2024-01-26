import streamlit as st
import numpy as np
import plotly.graph_objects as go    

def plot_vector(fig, v, o=np.array([0, 0, 0]), color='white'):
    fig.add_trace(go.Scatter3d(x=[o[0], v[0]], y=[o[1], v[1]], z=[o[2], v[2]],
                               mode='lines', line=dict(color=color)))

def plot_unit_cell(fig, V, vert_color='red', edge_color='white',s=5):
    fig.add_trace(go.Scatter3d(x=[x[0] for x in V], y=[x[1] for x in V], z=[x[2] for x in V], 
                               mode='markers', marker=dict(color=vert_color, size=s)))
    plot_vector(fig,V[1],V[0],color=edge_color)
    plot_vector(fig,V[2],V[0],color=edge_color)
    plot_vector(fig,V[3],V[0],color=edge_color)
    plot_vector(fig,V[1],V[4],color=edge_color)
    plot_vector(fig,V[1],V[5],color=edge_color)
    plot_vector(fig,V[4],V[7],color=edge_color)
    plot_vector(fig,V[2],V[4],color=edge_color)
    plot_vector(fig,V[2],V[4],color=edge_color)
    plot_vector(fig,V[2],V[6],color=edge_color)
    plot_vector(fig,V[3],V[6],color=edge_color)
    plot_vector(fig,V[3],V[5],color=edge_color)
    plot_vector(fig,V[7],V[6],color=edge_color)
    plot_vector(fig,V[7],V[5],color=edge_color)


def plot_crystal(a, b, c, beta, gamma, xtal, N,s=5):
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    I = np.identity(3)
    T = np.array([[a, 0, 0],
                  [b * np.cos(gamma), b * np.sin(gamma), 0],
                  [c * np.cos(beta), 0, c * np.sin(beta)]])
    P = np.dot(I, T)
    b1 = P[0]
    b2 = P[1]
    b3 = P[2]
    o = np.array([0, 0, 0])
    match xtal:
        case "Simple":
            V = [o, o + b1, o + b2, o + b3, o + b1 + b2, o + b1 + b3, o + b2 + b3, o + b1 + b2 + b3]
        case "Body":
            V = [o, o + b1, o + b2, o + b3, o + b1 + b2, o + b1 + b3, o + b2 + b3, o + b1 + b2 + b3, o + (b1 + b2 + b3)/2]
        case "Face":
            V = [o, o + b1, o + b2, o + b3, o + b1 + b2, o + b1 + b3, o + b2 + b3, o + b1 + b2 + b3,
                 o + (b1 + b2 )/2, o + (b1 + b3 )/2, o + (b2 + b3 )/2,
                 o + b3 + (b1 + b2 )/2, o + b2+(b1 + b3 )/2, o + b1+(b2 + b3 )/2]
        case "Basal":
            V = [o, o + b1, o + b2, o + b3, o + b1 + b2, o + b1 + b3, o + b2 + b3, o + b1 + b2 + b3,
                 o + (b1 + b2 )/2,  o + b3 + (b1 + b2 )/2]
        case _:
            V = [o, o + b1, o + b2, o + b3, o + b1 + b2, o + b1 + b3, o + b2 + b3, o + b1 + b2 + b3]

    fig = go.Figure()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                plot_unit_cell(fig, V + i * V[1] + k * V[2] + j * V[3] - (N*(b1+b2+b3)/2))
    return fig

def plot_hex_cell(fig,o,a,c,s=5):
    a2=a/2 #base of 30-60-90 with h=a
    a3=(a/2)*np.sqrt(3) #leg of 30-60-90 with h=a
    V=np.array([
        [0,0,0],
        [a,0,0],
        [a2,a3,0],
        [-a2,a3,0],
        [-a,0,0],
        [-a2,-a3,0],
        [a2,-a3,0], 
        [0,0,c],
        [a,0,c],
        [a2,a3,c],
        [-a2,a3,c],
        [-a,0,c],
        [-a2,-a3,c],
        [a2,-a3,c], 

    ])
    
    V=V+o
    fig.add_trace(go.Scatter3d(x=[x[0] for x in V], y=[x[1] for x in V], z=[x[2] for x in V], 
                               mode='markers', marker=dict(color='red', size=s)))
    
    plot_vector(fig,V[0],V[1])
    plot_vector(fig,V[0],V[2])
    plot_vector(fig,V[0],V[3])
    plot_vector(fig,V[0],V[4])
    plot_vector(fig,V[0],V[5])
    plot_vector(fig,V[0],V[6])
    
    plot_vector(fig,V[1],V[2])
    plot_vector(fig,V[2],V[3])
    plot_vector(fig,V[3],V[4])
    plot_vector(fig,V[4],V[5])
    plot_vector(fig,V[5],V[6])
    plot_vector(fig,V[6],V[1])
    
    plot_vector(fig,V[7],V[1+7])
    plot_vector(fig,V[7],V[2+7])
    plot_vector(fig,V[7],V[3+7])
    plot_vector(fig,V[7],V[4+7])
    plot_vector(fig,V[7],V[5+7])
    plot_vector(fig,V[7],V[6+7])
    
    plot_vector(fig,V[8],V[9])
    plot_vector(fig,V[9],V[10])
    plot_vector(fig,V[10],V[11])
    plot_vector(fig,V[11],V[12])
    plot_vector(fig,V[12],V[13])
    plot_vector(fig,V[13],V[8])
    
    plot_vector(fig,V[1],V[1+7])
    plot_vector(fig,V[2],V[2+7])
    plot_vector(fig,V[3],V[3+7])
    plot_vector(fig,V[4],V[4+7])
    plot_vector(fig,V[5],V[5+7])
    plot_vector(fig,V[6],V[6+7])
                
def plot_hex_crystal(a,c,N,s=5):
    o = np.array([0, 0, 0])
    b1=np.array([0,2*(a/2)*np.sqrt(3),0])
    b2=np.array([a,0,0])
    b3=np.array([0,0,c])
    center = np.array([(N-1)*a/2, (N-1)*a*np.sqrt(3)/2, N*c/2])
    fig = go.Figure()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                plot_hex_cell(fig,o+i*b1+j*b2+k*b3-center,a,c)
    return fig
# Example usage
# plot_crystal(1, 1, 1, 90, 90)
presets = {
    'Simple Cubic': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 90.0, 'gamma': 90.0, 'crystal_type': 'SC'},
    'Face Centered Cubic': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 90.0, 'gamma': 90.0, 'crystal_type': 'FCC'},
    'Body Centered Cubic': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 90.0, 'gamma': 90.0, 'crystal_type': 'BCC'},
    'Hexagonal Close Packed': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 60.0, 'gamma': 90.0, 'crystal_type': 'HCP'},
    'Simple Tetragonal': {'a': 1.0, 'b': 1.0, 'c': 2.0, 'beta': 90.0, 'gamma': 90.0, 'crystal_type': 'ST'},
    'Body Centered Tetragonal': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 60.0, 'gamma': 90.0, 'crystal_type': 'BCT'},
    'Rhombohedral': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 60.0, 'gamma': 60.0, 'crystal_type': 'RH'},
    'Simple Orthorhombic': {'a': 1.0, 'b': .75, 'c': .5, 'beta': 60.0, 'gamma': 90.0, 'crystal_type': 'SOR'},
    'Basal Orthorhombic': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 60.0, 'gamma': 90.0, 'crystal_type': 'BasalOR'},
    'Face Centered Orthorhombic': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 60.0, 'gamma': 90.0, 'crystal_type': 'FCOR'},
    'Body Centered Orthorhombic': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 60.0, 'gamma': 90.0, 'crystal_type': 'BCOR'},
    'Simple Monoclinic': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 60.0, 'gamma': 90.0, 'crystal_type': 'SM'},
    'Basal Monoclinic': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 60.0, 'gamma': 90.0, 'crystal_type': 'BasalM'},
    'Triclinic': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'beta': 60.0, 'gamma': 90.0, 'crystal_type': 'TC'}
}
def main():
    st.title("Bravais Lattice Visualizer")
    st.write("Choose the crystal size, unit cell type, and change the size/angle parameters in the side bar. Left click to orbit, scroll to zoom.\nSource: https://github.com/trilusa/bravais-lattice-viz")
    N = st.sidebar.number_input('Dimension of the Space Lattice', min_value=1, max_value=5, value=1, step=1)
    # s = st.sidebar.slider("Atom Size", min_value=1, max_value=20, value=, step=1)
    preset = st.sidebar.selectbox('Choose a Crystal Type', options=list(presets.keys()))
    print(preset)
    xtal = presets[preset]['crystal_type']
    maxval=2.0
    minval=0.1
    
    match xtal:
        case 'SC':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b=a
            c=a
            beta=90
            gamma=90
            xtal_type = "Simple"
        case 'FCC':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b=a
            c=a
            beta=90
            gamma=90
            xtal_type = "Face"
        case 'BCC':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b=a
            c=a
            beta=90
            gamma=90
            xtal_type = "Body"
        case 'HCP':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b=a
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta=60
            gamma=60
            fig = plot_hex_crystal(a, c, N)
        case 'ST':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b=a
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta=90
            gamma=90  
            xtal_type = "Simple"
        case 'BCT':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b=a
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta=90
            gamma=90  
            xtal_type = "Body"
        case 'RH':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b=a
            c=a
            beta = st.sidebar.slider('alpha', min_value=0.0, max_value=90.0, value=presets[preset]['beta'], key='beta')
            gamma = beta
            xtal_type = "Simple"
        case 'SOR': 
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b = st.sidebar.slider('b', min_value=minval, max_value=maxval, value=presets[preset]['b'], key='b')
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta=90
            gamma=90
            xtal_type = "Simple"
        case 'BasalOR':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b = st.sidebar.slider('b', min_value=minval, max_value=maxval, value=presets[preset]['b'], key='b')
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta=90
            gamma=90
            xtal_type = "Basal"            
        case 'FCOR':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b = st.sidebar.slider('b', min_value=minval, max_value=maxval, value=presets[preset]['b'], key='b')
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta=90
            gamma=90
            xtal_type = "Face" 
        case 'BCOR':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b = st.sidebar.slider('b', min_value=minval, max_value=maxval, value=presets[preset]['b'], key='b')
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta=90
            gamma=90
            xtal_type = "Body"
        case 'SM':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b = st.sidebar.slider('b', min_value=minval, max_value=maxval, value=presets[preset]['b'], key='b')
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta = st.sidebar.slider('beta', min_value=0.0, max_value=90.0, value=presets[preset]['beta'], key='beta')
            gamma = 90
            xtal_type = "Simple"
        case 'BasalM':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b = st.sidebar.slider('b', min_value=minval, max_value=maxval, value=presets[preset]['b'], key='b')
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta = st.sidebar.slider('beta', min_value=0.0, max_value=90.0, value=presets[preset]['beta'], key='beta')
            gamma = 90
            xtal_type = "Basal"
        case 'TC':
            a = st.sidebar.slider('a', min_value=minval, max_value=maxval, value=presets[preset]['a'], key='a')
            b = st.sidebar.slider('b', min_value=minval, max_value=maxval, value=presets[preset]['b'], key='b')
            c = st.sidebar.slider('c', min_value=minval, max_value=maxval, value=presets[preset]['c'], key='c')
            beta = st.sidebar.slider('beta', min_value=0.0, max_value=90.0, value=presets[preset]['beta'], key='beta')
            gamma = st.sidebar.slider('gamma', min_value=0.0, max_value=90.0, value=presets[preset]['gamma'], key='gamma')
            xtal_type = "Simple"
        case _:  
            print("Invalid Preset Selection")
                        
    if xtal != 'HCP':
        fig = plot_crystal(a, b, c, beta, gamma, xtal_type,N)

    for trace in fig.data:
        trace.hoverinfo = 'none'
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title='X', autorange=False),
    #         yaxis=dict(title='Y', autorange=False),
    #         zaxis=dict(title='Z', autorange=False),
    #         aspectmode='manual',
    #         aspectratio=dict(x=1, y=1, z=1)
    #     ),
    #     width=800,
    #     margin=dict(r=20, b=10, l=10, t=10),
    #     showlegend=False
    # )
    fig.update_layout(scene=dict(
                        xaxis=dict(
                            title='X',
                            autorange=False,
                            range=[-2*N, 2*N]  # Set your x-axis limits here
                        ),
                        yaxis=dict(
                            title='Y',
                            autorange=False,
                            range=[-2*N, 2*N]  # Set your y-axis limits here
                        ),
                        zaxis=dict(
                            title='Z',
                            autorange=False,
                            range=[-2*N, 2*N]  # Set your z-axis limits here
                        ),
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=1)  # Equal aspect ratio for all axes
                    ),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10),
                    showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()