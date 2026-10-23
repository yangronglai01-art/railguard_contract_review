"""Enterprise identity with the existing object grant boundary, stored in PG."""
from dataclasses import dataclass, field

from railguard.enterprise_identity import OIDC, IdentityError
from railguard.security import ROLES, AccessDenied, Principal, SecurityStore


@dataclass(frozen=True)
class OIDCPrincipal(Principal):
    access_token: str = field(default='',repr=False,compare=False)


class EnterpriseSecurityStore(SecurityStore):
    def __init__(self, repository, settings, *, oidc=None):
        super().__init__(repository,True)
        self.tenant=settings.enterprise_tenant_id
        self.oidc=oidc or OIDC(settings.oidc_introspection_url,settings.oidc_issuer,
                              settings.oidc_audience,settings.oidc_client_id,settings.oidc_client_secret)

    def initialize(self):
        self.repository.initialize()

    def identity(self, token):
        try:
            identity=self.oidc.resolve(token)
            roles=set(identity.roles)&ROLES
            if identity.tenant!=self.tenant or len(roles)!=1:raise IdentityError()
            return identity,next(iter(roles))
        except IdentityError:
            raise AccessDenied(401,'Invalid or expired enterprise identity') from None

    def authenticate(self, token):
        identity,role=self.identity(token)
        with self.repository._connect() as db:
            db.execute('''INSERT INTO rg_users VALUES(?,?,1,1) ON CONFLICT(subject)
                DO UPDATE SET role=excluded.role,active=1''',(identity.subject,role))
        return OIDCPrincipal(identity.subject,role,access_token=token)

    def _valid(self, conn, principal):
        try:
            identity,role=self.identity(principal.access_token)
            return identity.subject==principal.subject and role==principal.role
        except (AccessDenied,AttributeError):return False

    def issue_token(self,*args,**kwargs):
        raise ValueError('Enterprise identities are managed by the IdP')

    def upsert_user(self,*args,**kwargs):
        raise ValueError('Enterprise identities are managed by the IdP')
