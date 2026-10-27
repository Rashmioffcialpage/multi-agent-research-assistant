# Federated Learning

Federated learning trains a shared model across many clients (phones,
hospitals, edge devices) that each hold their own local data, without that
raw data ever leaving the client. Instead of centralizing data, the model
(or model updates) travels: a central server sends the current model to
clients, each client trains a few steps locally on its own data, and only
the resulting weight updates (not the data) are sent back and aggregated.

## FedAvg

The canonical algorithm, Federated Averaging (FedAvg), works in rounds.
Each round: the server broadcasts the current global model to a sampled
subset of clients; each client runs several epochs of local SGD on its own
data starting from that model; clients send their updated weights back;
the server averages them (typically weighted by how much local data each
client had) to produce the next global model. This is dramatically more
communication-efficient than sending raw gradients every step, since
clients do multiple local steps per round of communication.

## Why not just centralize the data

The motivation is usually regulatory or trust-based rather than purely
technical: healthcare data under HIPAA, financial data under various
banking regulations, or just a general unwillingness for users to have
their raw data (keystrokes, photos, health records) leave their device.
Federated learning lets multiple institutions collaboratively improve a
shared model — meaningful for domains like medical imaging, where any
single hospital's dataset is small and unrepresentative of the full
patient population, but pooling raw patient data across institutions is
legally and ethically fraught.

## Statistical heterogeneity (non-IID data)

The hardest practical problem in federated learning isn't the
communication protocol — it's that client data is rarely independent and
identically distributed. One hospital sees more of one condition; one
phone user's behavior patterns are unlike another's. Averaging model
updates computed on very different local distributions can produce a
global model that performs worse than training on any single client's
data alone (client drift). Approaches to mitigate this include adding a
proximal term that penalizes local updates for diverging too far from the
global model (FedProx), or personalization layers that stay local per
client while a shared backbone federates normally.

## Privacy beyond "the data didn't move"

Not sending raw data is necessary but not sufficient for real privacy
guarantees — model updates can still leak information about the training
data they were computed from (membership inference attacks, gradient
inversion). Differential privacy adds calibrated noise to updates before
aggregation, giving a formal, quantifiable privacy guarantee at the cost
of some accuracy. Secure aggregation is a complementary cryptographic
technique: the server only ever sees the *sum* of client updates, never
any individual client's update in isolation, which closes off a whole
class of attacks that target one client's contribution directly.
